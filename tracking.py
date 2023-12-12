from keypoint_annotation.angle_of_approach_labeler import \
    AngleOfApproachLabeler
from keypoint_annotation.dataset_maker import COCOMaker
from keypoint_annotation.keypoint_labeler import KPLabeler
from keypoint_annotation.slam_association import KeypointAssociation

if __name__ == "__main__":
    # KPLabeler("/home/matt/Datasets/towels/img").run()
    # KeypointAssociation("/home/matt/Datasets/towels").run()
    # AngleOfApproachLabeler("/home/matt/Datasets/towels").run()
    COCOMaker("/home/matt/Datasets/towels").run()

