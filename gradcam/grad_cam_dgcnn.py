import types

import numpy as np
import torch
from torch.autograd import Variable


class GradCamDGCNN:
    def __init__(self, model, counterfactual=False, normalize=True, disable_relu=True, use_cuda=True):
        #self.args = args
        #self.gradients = []
        #self.features = []
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.num_points = 2048
        self.num_cam = np.zeros(self.num_points)
        self.counterfactual = counterfactual
        self.normalize = normalize
        self.disable_relu = disable_relu

    def save_features(self, ctx, output):
        self.features = output
        #self.features_center_xyz.append(new_xyz)
        output.register_hook(self.save_gradients)

    def save_gradients(self, grad):
        self.gradients.append(grad)

    def calc_cam(self):
        grads_val = self.gradients[0].cpu().data.numpy()
        if self.counterfactual:
            grads_val = grads_val * -1
        target = self.features
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2))[0, :]
        #print(f"Weights - {weights.shape}")
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]
        if not self.disable_relu:
            cam = np.maximum(cam, 0) # ReLU
        #print(f"Cam - {cam.shape}")
        self.calculated_cam = cam

    def __call__(self, input, target_index=None):
        self.gradients = []
        self.features = []
        self.model.enco.save_features = types.MethodType(self.save_features, self.model)
        #grad_cam_input = input.transpose(2,1)
        #print(f"Input: {input.shape}")
        output = self.model(input)
        self.classifier_output = output
        
        index = np.argmax(output.cpu().data.numpy())
        if target_index is not None:
            index = target_index
        #print(f"Net Output {output.shape}") 
        #print(output) 
        #print(f"target is {index.item()}")
        #print()
        #print(f"Feature Map - {self.features[0].shape}")


        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)
        #print(f"target class score - {one_hot}")
        self.model.zero_grad()

        #print("performing backward pass....")
        one_hot.backward(retain_graph=True)
        #print(f"Gradients - {self.gradients[0].shape}")
        # Calculate Gradients


       
        self.calc_cam()
        cam_f = self.calculated_cam
        #print(f"Cam shape {cam_f.shape}")
        #print(cam_f)
        if self.normalize:
            # cam_f[np.isnan(cam_f)] = 0
            cam_f = cam_f - np.min(cam_f)
            cam_f = cam_f / np.max(cam_f)
        #print(cam_f)
        return cam_f