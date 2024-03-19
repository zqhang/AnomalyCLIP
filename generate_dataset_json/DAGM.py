import os
import json
import pandas as pd


class DAGMSolver(object):
    CLSNAMES = [
        'Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10'
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(Train={}, Test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['Train', 'Test']:
                cls_info = []
                x, y, mask_names_none= [], [], []
                img_dir = os.listdir(f'{cls_dir}/{phase}')
                
                mask_names = os.listdir(f'{cls_dir}/{phase}/Label')

                img_fpath_list = sorted([f
                                        for f in img_dir
                                        if f.endswith('.PNG')])
                gt_fpath_list = sorted([f
                            for f in mask_names
                            if f.endswith('.PNG')])

                img_exclude_list = [f.split("_")[0] + ".PNG" for f in gt_fpath_list]

                img_normal_fpath_list = list(set(img_fpath_list) - set(img_exclude_list))

                x.extend(img_normal_fpath_list + img_exclude_list)

                y.extend([0] * len(img_normal_fpath_list) + [1]* len(img_exclude_list))

                mask_names_none.extend([None] * len(img_normal_fpath_list) + gt_fpath_list)

                for idx, img_name in enumerate(x):
                    info_img = dict(
                        img_path=f'{cls_name}/{phase}/{img_name}',
                        mask_path=f'{cls_name}/{phase}/Label/{mask_names_none[idx]}' if mask_names_none[idx] != None else '',
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if y[idx] == 1 else 0,
                    )
                    cls_info.append(info_img)
                    if phase == 'Test':
                        if y[idx] == 1:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)



if __name__ == '__main__':
    runner = DAGMSolver(root='/remote-home/iot_zhouqihang/data/DAGM_KaggleUpload')
    runner.run()
