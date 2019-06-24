import  torch
import sys, os, argparse

from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL



class OcrdAnybaseocrDewarper():

    def __init__(self, param):
        self.param = param

    def dewarping(self, tmp, dest):
        os.system("python pix2pixHD/test.py --dataroot %s --checkpoints_dir ./ --name models --results_dir %s --label_nc 0 --no_instance --no_flip --resize_or_crop none --n_blocks_global 10 --n_local_enhancers 2 --gpu_ids %s --loadSize %d --fineSize %d --resize_or_crop %s" % (os.path.dirname(tmp), dest, self.param['gpu_id'], self.param['resizeHeight'], self.param['resizeWidth'], self.param['imgresize']))



def main():
    parser = argparse.ArgumentParser("""
    Image dewarping using pix2pixHD.

            python ocrd_anyBaseOCR_dewarp.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)

    This is a compute-intensive dewarping method that works on degraded
    and historical book pages.
    """)

    parser.add_argument('-p', '--parameter', type=str, help="Parameter file location")
    parser.add_argument('-w', '--work', type=str, help="Working directory location", default=".")
    parser.add_argument('-I', '--Input', default=None, help="Input directory")
    parser.add_argument('-O', '--Output', default=None, help="output directory")
    parser.add_argument('-m', '--mets', default=None, help="METs input file")
    parser.add_argument('-o', '--OutputMets', default=None, help="METs output file")
    parser.add_argument('-g', '--group', default=None, help="METs image group id")
    args = parser.parse_args()

    # Read parameter values from json file
    param = {}
    if args.parameter:
        with open(args.parameter, 'r') as param_file:
            param = json.loads(param_file.read())
    param = parse_params_with_defaults(param, OCRD_TOOL['tools']['ocrd-anybaseocr-dewarp']['parameters'])

    if not args.mets or not args.Input or not args.Output or not args.work:
        parser.print_help()
        print("Example: ocrd_anyBaseOCR_dewarp.py -m (mets input file path) -I (input-file-grp name) -O (output-file-grp name) -w (Working directory)")
        sys.exit(0)

    if args.work:
        if not os.path.exists(args.work):
            os.mkdir(args.work)

    if torch.cuda.is_available():
        count_cuda_device = torch.cuda.device_count()
        for i in range(count_cuda_device):
            ss = str(i) if i == 0 else ss+','+str(i)
        os.system("export CUDA_VISIBLE_DEVICES=%s" % ss)
        print("export CUDA_VISIBLE_DEVICES=%s" % ss)
    else:
        print("Your system has no CUDA installed. No GPU detected.")
        sys.exit(0)

    dewarper = OcrdAnybaseocrDewarper(param)
    files = parseXML(args.mets, args.Input)
    fnames = []
    img_tmp_dir = "work_dir/test_A"

    for i, fname in enumerate(files):
        base = str(fname).split('.')[0]
        img_dir = os.path.dirname(str(fname))
        os.system("mkdir -p %s" % img_tmp_dir)
        os.system("cp %s %s" % (str(fname), os.path.join(img_tmp_dir, os.path.basename(str(fname)))))
        fnames.append(base + '.dw.png')
    dewarper.dewarping(img_tmp_dir, img_dir)
    os.system("rm -r %s" % img_tmp_dir)
    write_to_xml(fnames, args.mets, args.Output, args.OutputMets, args.work)
