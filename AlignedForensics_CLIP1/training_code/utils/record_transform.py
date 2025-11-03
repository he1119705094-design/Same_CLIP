import torch
import torchvision.transforms as transforms

class TransformRecorder:
    def __init__(self, transform_pipeline):
        self.transform_pipeline = transform_pipeline
        self.applied_transforms = []

    def __call__(self, sample):
        # Initialize an empty list to store the applied transformations
        self.applied_transforms = []
        
        # Apply each transformation and store the lambda function
        for transform in self.transform_pipeline.transforms:
            # Store the transform's state in a lambda function, assuming each transform has a __call__ method
            transform_func = lambda x, t=transform: t(x)
            sample = transform_func(sample)
            self.applied_transforms.append(transform_func)
        
        # Return the transformed sample and the applied transformations
        return sample, self.applied_transforms

    def apply_same_operations(self, sample):
        for transform_func in self.applied_transforms:
            sample = transform_func(sample)
        return sample
