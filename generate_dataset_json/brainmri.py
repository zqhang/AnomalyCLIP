import os
import json


class IsbiSolver(object):
    CLSNAMES = ['brain']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/brain_tumor_dataset'
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}')
                for specie in species:
                    is_abnormal = True if specie not in ['no'] else False
                    img_names = os.listdir(f'{cls_dir}/{specie}')
                    img_names.sort()
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_dir}/{specie}/{img_name}',
                            cls_name=cls_name,
                            mask_path="",
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = IsbiSolver(root='/remote-home/iot_zhouqihang/data/BrainMRI')
    runner.run()
