# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import torchvision
import numpy as np 
import torch.nn.functional as F
import cv2
from tqdm import tqdm


sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss
from helper_train import getrenderpip, getmodel, getloss, controlgaussians, reloadhelper, trbfunction, setgtisint8, getgtisint8
from thirdparty.gaussian_splatting.scene import Scene
from argparse import Namespace
from thirdparty.gaussian_splatting.helper3dg import getparser, getrenderparts


def train(dataset, opt, pipe, saving_iterations, debug_from, densify=0, duration=50, rgbfunction="rgbv1", rdpip="v2"):
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    first_iter = 0
    render, GRsetting, GRzer = getrenderpip(rdpip)

    print("use model {}".format(dataset.model))
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    
    gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
    gaussians.trbfslinit = -1*opt.trbfslinit # 
    gaussians.preprocesspoints = opt.preprocesspoints 
    gaussians.addsphpointsscale = opt.addsphpointsscale 
    gaussians.raystart = opt.raystart


    rbfbasefunction = trbfunction
    scene = Scene(dataset, gaussians, duration=duration, loader=dataset.loader)
    

    currentxyz = gaussians._xyz 
    maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
    minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
     

    if os.path.exists(opt.prevpath):
        print("load from " + opt.prevpath)
        reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz)
   


    maxbounds = [maxx, maxy, maxz]
    minbounds = [minx, miny, minz]


    gaussians.training_setup(opt)
    
    numchannel = 9 

    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(numchannel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    #if freeze != 1:
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    flag = 0
    flagtwo = 0
    depthdict = {}

    if opt.batch > 1:
        traincameralist = scene.getTrainCameras().copy()
        traincamdict = {}
        for i in range(duration): # 0 to 4, -> (0.0, to 0.8)
            traincamdict[i] = [cam for cam in traincameralist if cam.timestamp == i/duration]
    
    
    if gaussians.ts is None :
        H,W = traincameralist[0].image_height, traincameralist[0].image_width
        gaussians.ts = torch.ones(1,1,H,W).cuda()

    scene.recordpoints(0, "start training")

                                                            
    flagems = 0  
    emscnt = 0
    lossdiect = {}
    ssimdict = {}
    depthdict = {}
    validdepthdict = {}
    emsstartfromiterations = opt.emsstart   




    ########################### Load Magnitudes ##############################
        # load the magnitudes of all the cameras
    # first get names of all the cameras
    cameras = []
    path1 = dataset.source_path
    # clip the colmap_0/ from the path
    path1 = path1.split("colmap_0")[0]
    for folder in os.listdir( path1):
        if folder.startswith("cam") and os.path.isdir(os.path.join(path1, folder)):
            cameras.append(folder)

    print(f"Number of cameras: {len(cameras)}")

    motion_magnitudes = {}
    # load the magnitudes of all the cameras
    # for camera in cameras:
    #     file_path = os.path.join(path1, camera, "magnitudes.npy")
    #     magnitudes = np.load(file_path)
    #     motion_magnitudes[camera] = magnitudes

    # load the magnitudes of all the cameras convert to tensor and move to cuda
    for camera in cameras:
        file_path = os.path.join(path1, camera, "magnitudes.npy")
        magnitudes = np.load(file_path)
        magnitudes = torch.tensor(magnitudes).cuda()
        motion_magnitudes[camera] = magnitudes

    # max magnitude dictionary
    max_magnitudes = {}
    list_of_all_max_magnitudes = []
    # get image size
    image_size = traincameralist[0].original_image.size()
    H, W = image_size[1], image_size[2]
    # ite4rate over all patches and get the magnitudes
    for camera in cameras:
        for timestamp in range(1,duration):
            magnitudes = motion_magnitudes[camera][timestamp-1]
            # get the patches
            kh, kw = 16, 16 # kernel size
            dh, dw = 16, 16 # stride
            idealh, idealw = int(H / dh  + 1) * kw, int(W / dw + 1) * kw # compute padding
            magnitudes = torch.nn.functional.pad(magnitudes, (0, idealw - magnitudes.shape[1], 0, idealh - magnitudes.shape[0]), mode='constant', value=0)
            patches = magnitudes.unfold(0, kh, dh).unfold(1, kw, dw)
            # sum the magnitudes in each patch
            patchessum = patches.mean(dim=(2,3))
            # get max magnitude from patches
            max_magnitude = patchessum.max()
            # save the max magnitude in the dictionary
            if camera not in max_magnitudes:
                max_magnitudes[camera] = {}
                # for timestamp 0 the max magnitude is 0
                max_magnitudes[camera][0] = torch.tensor(0).cuda()
            max_magnitudes[camera][timestamp] = max_magnitude
            list_of_all_max_magnitudes.append(max_magnitude)

    # divide the list of all max magnitudes into 3 parts DESCENDING ORDER
    list_of_all_max_magnitudes = sorted(list_of_all_max_magnitudes, reverse=True)
    N = len(list_of_all_max_magnitudes)
    
    threshold_for_High = list_of_all_max_magnitudes[int(N*0.05)]
    threshold_for_Medium = list_of_all_max_magnitudes[int(N*0.15)]

    # get the high, medium and low cameras frames from max_magnitudes dictionary
    high_cameras = {}
    medium_cameras = {}
    low_cameras = {}

    for camera in cameras:
        for timestamp in range(duration):
            max_magnitude = max_magnitudes[camera][timestamp]
            if max_magnitude >= threshold_for_High:
                if timestamp not in high_cameras:
                    high_cameras[timestamp] = []
                high_cameras[timestamp].append(camera)
            elif max_magnitude >= threshold_for_Medium:
                if timestamp not in medium_cameras:
                    medium_cameras[timestamp] = []
                medium_cameras[timestamp].append(camera)
            else:
                if timestamp not in low_cameras:
                    low_cameras[timestamp] =  []
                low_cameras[timestamp].append(camera)

    # save the high, medium and low cameras in a json file
    with open(os.path.join(dataset.model_path, "high_cameras.json"), 'w') as f:
        json.dump(high_cameras, f)
    with open(os.path.join(dataset.model_path, "medium_cameras.json"), 'w') as f:
        json.dump(medium_cameras, f)
    with open(os.path.join(dataset.model_path, "low_cameras.json"), 'w') as f:
        json.dump(low_cameras, f)





    with torch.no_grad():
        timeindex = 0 # 0 to 49
        viewpointset = traincamdict[timeindex]
        for viewpoint_cam in viewpointset:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer)
            
            _, depthH, depthW = render_pkg["depth"].shape
            borderH = int(depthH/2)
            borderW = int(depthW/2)

            midh =  int(viewpoint_cam.image_height/2)
            midw =  int(viewpoint_cam.image_width/2)
            
            depth = render_pkg["depth"]
            slectemask = depth != 15.0 

            validdepthdict[viewpoint_cam.image_name] = torch.median(depth[slectemask]).item()   
            depthdict[viewpoint_cam.image_name] = torch.amax(depth[slectemask]).item() 
    
    if densify == 1 or  densify == 2: 
        zmask = gaussians._xyz[:,2] < 4.5  
        gaussians.prune_points(zmask) 
        torch.cuda.empty_cache()


    selectedlength = 2
    lasterems = 0 
    gtisint8 = getgtisint8()

    for iteration in range(first_iter, opt.iterations + 1):        
        if iteration ==  opt.emsstart:
            flagems = 1 # start ems

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.train()

        if opt.batch > 1:
            gaussians.zero_gradient_cache()
            # timeindex = randint(0, duration-1) # 0 to 49
            # viewpointset = traincamdict[timeindex]
            # camindex = random.sample(viewpointset, opt.batch)

            prob_time = iteration%3
            if prob_time == 0:
                timeindex = random.choice(list(low_cameras.keys()))
                viewpointset = low_cameras[timeindex]
            elif prob_time == 1:
                timeindex = random.choice(list(medium_cameras.keys()))
                viewpointset = medium_cameras[timeindex]
            else:
                timeindex = random.choice(list(high_cameras.keys()))
                viewpointset = high_cameras[timeindex]
            ###########################################################################
            # if iteration < emsstartfromiterations:
            #     # choose random timestamp from low_cameras dictionary
            #     timeindex = random.choice(list(low_cameras.keys()))
            #     viewpointset = low_cameras[timeindex]
            # elif iteration < emsstartfromiterations * 2:
            #     # 50% of the time choose from low_cameras and 40% from medium_cameras and 10% from high_cameras
            #     random_number = random.random()
            #     if random_number < 0.5:
            #         timeindex = random.choice(list(low_cameras.keys()))
            #         viewpointset = low_cameras[timeindex]
            #     elif random_number < 0.9:
            #         timeindex = random.choice(list(medium_cameras.keys()))
            #         viewpointset = medium_cameras[timeindex]
            #     else:
            #         timeindex = random.choice(list(high_cameras.keys()))
            #         viewpointset = high_cameras[timeindex]
            # else:
            #     # 30% of the time choose from low_cameras and 30% from medium_cameras and 40% from high_cameras
            #     random_number = random.random()
            #     if random_number < 0.3:
            #         timeindex = random.choice(list(low_cameras.keys()))
            #         viewpointset = low_cameras[timeindex]
            #     elif random_number < 0.6:
            #         timeindex = random.choice(list(medium_cameras.keys()))
            #         viewpointset = medium_cameras[timeindex]
            #     else:
            #         timeindex = random.choice(list(high_cameras.keys()))
            #         viewpointset = high_cameras[timeindex]
            
            # get the camera index and timestamp from traincamdict
            tempviewpointset = traincamdict[timeindex]
            # remove those which are not in viewpointset
            tempset = []
            for temp in tempviewpointset:
                if temp.image_name in viewpointset:
                    tempset.append(temp)
            
            viewpointset = tempset
            if len(viewpointset) < opt.batch:
                camindex = viewpointset
            else:
                camindex = random.sample(viewpointset, opt.batch)

#############################################################################################################
            for i in range(len(camindex)): 
                viewpoint_cam = camindex[i]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer)
                image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg) 
                
                if gtisint8:
                    gt_image = viewpoint_cam.original_image.cuda().float()/255.0
                else:
                    # cast float on cuda will introduce gradient, so cast first then to cuda. at the cost of i/o
                    gt_image = viewpoint_cam.original_image.float().cuda()  
                if opt.gtmask: # for training with undistorted immerisve image, masking black pixels in undistorted image. 
                    mask = torch.sum(gt_image, dim=0) == 0
                    mask = mask.float()
                    image = image * (1- mask) +  gt_image * (mask)


                if opt.reg == 2:
                    Ll1 = l2_loss(image, gt_image)
                    loss = Ll1
                elif opt.reg == 3:
                    Ll1 = rel_loss(image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii)

                if flagems == 1:
                    if viewpoint_cam.image_name not in lossdiect:
                        lossdiect[viewpoint_cam.image_name] = loss.item()
                        ssimdict[viewpoint_cam.image_name] = ssim(image.clone().detach(), gt_image.clone().detach()).item()
                
                loss.backward()
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none = True)# 



            if flagems == 1 and len(lossdiect.keys()) == len(viewpointset):
                # sort dict by value
                orderedlossdiect = sorted(ssimdict.items(), key=lambda item: item[1], reverse=False) # ssimdict lossdiect
                flagems = 2
                selectviewslist = []
                selectviews = {}
                for idx, pair in enumerate(orderedlossdiect):
                    viewname, lossscore = pair
                    ssimscore = ssimdict[viewname]
                    if ssimscore < 0.91: # avoid large ssim
                        selectviewslist.append((viewname, "rk"+ str(idx) + "_ssim" + str(ssimscore)[0:4]))
                if len(selectviewslist) < 2 :
                    selectviews = []
                else:
                    selectviewslist = selectviewslist[:2]
                    for v in selectviewslist:
                        selectviews[v[0]] = v[1]

                selectedlength = len(selectviews)

            iter_end.record()
            gaussians.set_batch_gradient(opt.batch)
             # note we retrieve the correct gradient except the mask
        else:
            raise NotImplementedError("Batch size 1 is not supported")
        
##################################################################################
        # create 2 folders medium and high
        # if not os.path.exists(os.path.join(dataset.model_path, "medium")):
        #     os.makedirs(os.path.join(dataset.model_path, "medium"))
        # if not os.path.exists(os.path.join(dataset.model_path, "high")):
        #     os.makedirs(os.path.join(dataset.model_path, "high"))

        # if prob_time == 1:
        #     path = os.path.join(dataset.model_path, "medium")
        #     # save training images
        #     torchvision.utils.save_image(image, os.path.join(path, "render" + str(iteration)+"_" + str(timeindex)+"_"+str(camindex[-1].image_name) +".png"))
        # elif prob_time == 2:
        #     path = os.path.join(dataset.model_path, "high")
        #     # save training images
        #     torchvision.utils.save_image(image, os.path.join(path, "render" + str(iteration) + "_" + str(timeindex)+"_"+str(camindex[-1].image_name) +".png"))
############################################################################################





        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            # Densification and pruning here
            
            if iteration < opt.densify_until_iter :
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            flag = controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag,  traincamerawithdistance=None, maxbounds=maxbounds,minbounds=minbounds)
           
            # guided sampling step
            if iteration > emsstartfromiterations and flagems == 2 and emscnt < selectedlength and viewpoint_cam.image_name in selectviews and (iteration - lasterems > 100): #["camera_0002"] :#selectviews :  #["camera_0002"]:
                selectviews.pop(viewpoint_cam.image_name) # remove sampled cameras
                emscnt += 1
                lasterems = iteration
                ssimcurrent = ssim(image.detach(), gt_image.detach()).item()
                scene.recordpoints(iteration, "ssim_" + str(ssimcurrent))
                # some scenes' strcture is already good, no need to add more points
                if ssimcurrent < 0.88:
                    imageadjust = image /(torch.mean(image)+0.01) # 
                    gtadjust = gt_image / (torch.mean(gt_image)+0.01)
                    diff = torch.abs(imageadjust   - gtadjust)
                    diff = torch.sum(diff,        dim=0) # h, w
                    diff_sorted, _ = torch.sort(diff.reshape(-1)) 
                    numpixels = diff.shape[0] * diff.shape[1]
                    threshold = diff_sorted[int(numpixels*opt.emsthr)].item()
                    outmask = diff > threshold#  
                    kh, kw = 16, 16 # kernel size
                    dh, dw = 16, 16 # stride
                    idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute padding  
                    outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)
                    patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummypatch = torch.ones_like(patches)
                    patchessum = patches.sum(dim=(2,3)) 
                    patchesmusk = patchessum  >  kh * kh * 0.85
                    patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                    patches = dummypatch * patchesmusk

                    depth = render_pkg["depth"]
                    depth = depth.squeeze(0)
                    idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute padding for depth

                    depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                    depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummydepthpatches =  torch.ones_like(depthpaches)
                    a,b,c,d = depthpaches.shape
                    depthpaches = depthpaches.reshape(a,b,c*d)
                    mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                    depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                    unfold_depth_shape = dummydepthpatches.size()
                    output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                    output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                    patches_depth_orig = depthpaches.view(unfold_depth_shape)
                    patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                    patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # 1 for error, 0 for no error

                    depth = patches_depth[:render_pkg["depth"].shape[1], :render_pkg["depth"].shape[2]]
                    depth = depth.unsqueeze(0)


                    midpatch = torch.ones_like(patches)
      

                    for i in range(0, kh,  2):
                        for j in range(0, kw, 2):
                            midpatch[:,:, i, j] = 0.0  
   
                    centerpatches = patches * midpatch

                    unfold_shape = patches.size()
                    patches_orig = patches.view(unfold_shape)
                    centerpatches_orig = centerpatches.view(unfold_shape)

                    output_h = unfold_shape[0] * unfold_shape[2]
                    output_w = unfold_shape[1] * unfold_shape[3]
                    patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                    centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                    centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                    centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back
                    
                    errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                    errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                    H, W = centermask.shape

                    offsetH = int(H/10)
                    offsetW = int(W/10)

                    centermask[0:offsetH, :] = 0.0
                    centermask[:, 0:offsetW] = 0.0

                    centermask[-offsetH:, :] = 0.0
                    centermask[:, -offsetW:] = 0.0


                    depth = render_pkg["depth"]
                    depthmap = torch.cat((depth, depth, depth), dim=0)
                    invaliddepthmask = depth == 15.0

                    pathdir = scene.model_path + "/ems_" + str(emscnt-1)
                    if not os.path.exists(pathdir): 
                        os.makedirs(pathdir)
                    
                    depthmap = depthmap / torch.amax(depthmap)
                    invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()  


                    torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "gt" + str(iteration) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(pathdir,  "render" + str(iteration) + ".png"))
                    torchvision.utils.save_image(depthmap, os.path.join(pathdir,  "depth" + str(iteration) + ".png"))
                    torchvision.utils.save_image(invalideptmap, os.path.join(pathdir,  "indepth" + str(iteration) + ".png"))
                    

                    badindices = centermask.nonzero()
                    diff_sorted , _ = torch.sort(depth.reshape(-1)) 
                    N = diff_sorted.shape[0]
                    mediandepth = int(0.7 * N)
                    mediandepth = diff_sorted[mediandepth]

                    depth = torch.where(depth>mediandepth, depth,mediandepth )

                  
                    totalNnewpoints = gaussians.addgaussians(badindices, viewpoint_cam, depth, gt_image, numperay=opt.farray,ratioend=opt.rayends,  depthmax=depthdict[viewpoint_cam.image_name], shuffle=(opt.shuffleems != 0))

                    gt_image = gt_image * errormask
                    image = render_pkg["render"] * errormask

                    scene.recordpoints(iteration, "after addpointsbyuv")

                    torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "maskedudgt" + str(iteration) + ".png"))
                    torchvision.utils.save_image(image, os.path.join(pathdir,  "maskedrender" + str(iteration) + ".png"))
                    visibility_filter = torch.cat((visibility_filter, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                    visibility_filter = visibility_filter.bool()
                    radii = torch.cat((radii, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                    viewspace_point_tensor = torch.cat((viewspace_point_tensor, torch.zeros(totalNnewpoints, 3).cuda(0)), dim=0)


                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                
                gaussians.optimizer.zero_grad(set_to_none = True)



if __name__ == "__main__":
    

    args, lp_extract, op_extract, pp_extract = getparser()
    setgtisint8(op_extract.gtisint8)
    train(lp_extract, op_extract, pp_extract, args.save_iterations, args.debug_from, densify=args.densify, duration=args.duration, rgbfunction=args.rgbfunction, rdpip=args.rdpip)

    # All done
    print("\nTraining complete.")
