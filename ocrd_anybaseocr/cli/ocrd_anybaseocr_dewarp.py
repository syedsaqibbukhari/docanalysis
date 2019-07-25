import  torch
import sys, os

from ..utils import parseXML, write_to_xml, print_info, parse_params_with_defaults, print_error
from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml,parse
from ocrd_models.ocrd_page_generateds import BorderType
import shutil
from pathlib import Path
from PIL import Image
import ocrolib



class OcrdAnybaseocrDewarper(Processor):

        
        def __init__(self, *args, **kwargs):          
            kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-dewarp']
            kwargs['version'] = OCRD_TOOL['version']
            super(OcrdAnybaseocrDewarper, self).__init__(*args, **kwargs)
                
        def check_cuda(self):          
            if torch.cuda.is_available():                        
                return True
            else:
                return False

        def check_pix2pix(self):               
            abs_path = Path(self.parameter['pix2pixHD']).absolute()
            work_dir = Path(self.workspace.directory + "/pix2pixHD")            
            if Path(abs_path).exists():                
                return abs_path
            else:
                if Path(work_dir).exists():                    
                    return work_dir
                else:                    
                    return None

        def crop_image(self,image_path,crop_region):            
            img = Image.open(image_path)
            cropped = img.crop(crop_region)            
            return cropped
            
            

        def process(self):            
            if self.check_cuda():
                path = self.check_pix2pix()
                if not (path is None):
                    for (n, input_file) in enumerate(self.input_files):                                                                        
                        local_input_file = self.workspace.download_file(input_file)
                        pcgts = parse(local_input_file.url, silence=True)
                        image_coords = pcgts.get_Page().get_Border().get_Coords().points.split()
                        fname = pcgts.get_Page().imageFilename
                                                
                        # Get page Co-ordinates
                        min_x, min_y = image_coords[0].split(",")
                        max_x,max_y = image_coords[2].split(",")
                        img_tmp_dir = "OCR-D-IMG/test_A"                                                       
                        img_dir = os.path.dirname(str(fname))
                        # Path of pix2pixHD                        
                        Path(img_tmp_dir).mkdir(parents=True, exist_ok=True)

                        crop_region = int(min_x), int(min_y), int(max_x), int(max_y)                        
                        cropped_img = self.crop_image(fname, crop_region)

                        base,_ = ocrolib.allsplitext(fname)
                        filename = base.split("/")[-1] + ".png"
                        cropped_img.save(img_tmp_dir + "/" + filename)                        
                        #os.system("cp %s %s" % (str(fname), os.path.join(img_tmp_dir, os.path.basename(str(fname)))))                        
                        #os.system("mkdir -p %s" % img_tmp_dir)                        
                        #os.system("cp %s %s" % (str(fname), os.path.join(img_tmp_dir, os.path.basename(str(fname)))))                    
                    os.system("python " + str(path) + "/test.py --dataroot %s --checkpoints_dir ./ --name models --results_dir %s --label_nc 0 --no_instance --no_flip --resize_or_crop none --n_blocks_global 10 --n_local_enhancers 2 --gpu_ids %s --loadSize %d --fineSize %d --resize_or_crop %s" % (os.path.dirname(img_tmp_dir), img_dir, self.parameter['gpu_id'], self.parameter['resizeHeight'], self.parameter['resizeWidth'], self.parameter['imgresize']))
                    synthesized_image = filename.split(".")[0] + "_synthesized_image.jpg"
                    pix2pix_img_dir = img_dir + "/models/test_latest/images/"
                    dewarped_image = Path(pix2pix_img_dir + synthesized_image)  
                    if(dewarped_image.is_file()):
                        shutil.copy(dewarped_image, img_dir + "/"+ filename.split(".")[0] + "_dw.jpg")                      
                    
                    if(Path(img_tmp_dir).is_dir()):
                        shutil.rmtree(img_tmp_dir)
                    if(Path(img_dir + "/models").is_dir()):
                        shutil.rmtree(img_dir + "/models")
                    
                else:
                    print(""" Please check if pix2pixHD is downloaded in path docanalysis/ocrd_anybaseocr 
                        If not, you can clone the repository from the following url:
                            https://github.com/NVIDIA/pix2pixHD
                        Alternatively, you can change the pix2pixHD path in ocrd-json file                            
                        """)
             
            else:                
                print("Your system has no CUDA installed. No GPU detected.")
                sys.exit(0)