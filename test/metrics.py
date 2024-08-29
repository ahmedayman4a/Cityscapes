import torch
import torch.nn.functional as F
import pytorch_toolbelt.losses as L

class Metric:
    def __init__(self, eps=1e-7) -> None:
        self.eps = eps
        
    def calculate(self, output, target):
        pass
    
class IOUScore(Metric):
    def __init__(self, num_classes, ignore_index_array = [0], eps=1e-7) -> None:
        super().__init__(eps)
        self.num_classes = num_classes
        self.ignore_index_array = ignore_index_array
        
    def calculate(self, output:torch.Tensor, target:torch.Tensor):
        """
        Calculate Intersection over Union (IoU) for each class.

        Arguments:
            output (torch.Tensor): One hot model output (NxCxHxW)
            target (torch.Tensor): Ground truth label (NxHxW)

        Returns:
            float: The mean IoU
            list: A list of tuples where each tuple is (class ID, IoU)
        """
        iou_list = []
        gt_classes = set(torch.unique(target).tolist())  
        output = output.argmax(dim=1)

        for cls in range(self.num_classes):
            if self.ignore_index_array is not None and cls in self.ignore_index_array:
                continue  # Skip the ignored class

            # Calculate intersection and union
            intersection = torch.sum((output == cls) & (target == cls)).item()
            union = torch.sum((output == cls) | (target == cls)).item()

            # Compute IoU for the class
            iou = intersection / union if union > 0 else 1
            iou_list.append((cls, iou))  # Store class ID with IoU value

        # Filter IoU list to only include classes present in ground truth and not ignored
        valid_iou_list = [(cls, iou) for cls, iou in iou_list if cls in gt_classes]

        # Compute mean IoU, excluding classes that are not present in ground truth
        if len(valid_iou_list) > 0:
            mean_iou = sum(iou for _, iou in valid_iou_list) / len(valid_iou_list)
        else:
            mean_iou = 0.0

        return mean_iou, iou_list
    
class DiceScore(Metric):
    def __init__(self,ignore_index=0, eps=1e-7) -> None:
        super().__init__(eps)
        self.ignore_index = ignore_index
        self.loss_function = L.DiceLoss(mode='multiclass', ignore_index=ignore_index, eps=eps)
        
    def calculate(self, output, target):
        return 1 - self.loss_function(output, target).item()
    
