from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def eval(res_path):
    annotation_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'
    results_file = f'/kaggle/working/{res_path}'

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')