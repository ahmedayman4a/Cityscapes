import torch
from data_handling.data_reader import labels
from test.metrics import DiceScore, IOUScore
from tqdm import tqdm
import numpy as np

trainId_to_name = {label.trainId: ("background" if label.trainId == 0 else label.name) for label in labels}

class TestModel:
    def __init__(self, model, dataloader, use_gpu, ignore_index=0, num_classes=20):
        self.model = model
        self.dataloader = dataloader
        self.use_gpu = use_gpu
        self.dice_score = DiceScore(ignore_index=ignore_index)
        self.iou_score = IOUScore(num_classes=num_classes, ignore_index_array=[ignore_index])
        
    def __call__(self):
        self.model.eval()
        
        all_dice_scores = []
        all_jaccard_scores = []
        class_iou_dict = {cls: [] for cls in range(20)} 
        
        with torch.no_grad():
            for iter, (inputs, labels) in tqdm(enumerate(self.dataloader)):
                if self.use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self.model(inputs)
                all_dice_scores.append(self.dice_score.calculate(outputs, labels))
                mean_iou, iou_scores = self.iou_score.calculate(outputs, labels)
                all_jaccard_scores.append(mean_iou)
                # Update class IoU dictionary
                for cls, iou in iou_scores:
                    class_iou_dict[cls].append(iou)
                
                if (iter+1) % 10 == 0:
                    print(f'Finished iteration {iter+1}, Calculating averaged metrics...')
                    break
        
        avg_dice = np.mean(all_dice_scores, axis=0)
        avg_jaccard = np.mean(all_jaccard_scores, axis=0)
        
        # Calculate and print the average IoU for each class
        print(f'\nAverage IoU for Each Class:')
        print('-' * 40)
        for cls, iou_list in class_iou_dict.items():
            if len(iou_list) > 0:
                avg_class_iou = np.mean(iou_list)
                class_name = trainId_to_name.get(cls, f'Class {cls}')
                print(f'{class_name:<20} {avg_class_iou:.4f}')
        
        print(f'\nAverage Metrics:')
        print(f'{"Metric":<20}{"Score":<10}')
        print('-' * 40)
        print(f'{"Dice Score":<20}{avg_dice:<10.4f}')
        print(f'{"mIOU":<20}{avg_jaccard:<10.4f}')
        
        return avg_dice, avg_jaccard, class_iou_dict