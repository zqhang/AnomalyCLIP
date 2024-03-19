import os
import json

import os
import json


class MpddSolver(object):
    CLSNAMES = ['chest']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}'
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}')
                for specie in species:
                    is_abnormal = True if specie not in ['NORMAL'] else False
                    img_names = os.listdir(f'{cls_dir}/{specie}/')

                    img_names.sort()

                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{specie}/{img_name}',
                            mask_path="",
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = MpddSolver(root='/remote-home/iot_zhouqihang/data/COVID-19_Radiography_Dataset')
    runner.run()
