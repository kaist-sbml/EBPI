import fitz
import io
import os
import pandas as pd
import requests
import shutil
import tarfile
import time
from Bio import Entrez
from PIL import Image


def bulkdownload(args):
    headers= dict()
    headers['user_agent']= args.header
    pmc_name_dict=dict()
    abs_path= os.path.dirname(__file__)
    pmc_list1= pd.read_csv(abs_path+'/oa_non_comm_use_pdf.csv')
    pmc_list2= pd.read_csv(abs_path+'/oa_comm_use_file_list.csv')

    for name in list(pmc_list1['File']):
        pmc_name_dict[name.split('.')[1]]= 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/'+ name

    for name in list(pmc_list2['File']):
        pmc_name_dict[name.split('/')[-1].strip('.tar.gz')] = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/'+ name

    Entrez.email = args.email
    terms= args.metabolite+' AND metabolic engineering'
    handle = Entrez.esearch(db="pmc", term=terms, retmax=args.len)
    records = Entrez.read(handle)['IdList']
    i=0
    result_list= []
    for record in records:
        if 'PMC'+ record in pmc_name_dict:
            result_list.append(pmc_name_dict['PMC'+record])

    if not os.path.exists(abs_path+'/output_file/'+args.metabolite+'/0'):
        os.makedirs(abs_path+'/output_file/'+args.metabolite+'/0')
    
    print("The number of papers to download: %i"%len(result_list))
    
    for url in result_list:
        print("URL: %s"%url)
        file_name = abs_path+'/output_file/'+ args.metabolite+'/'+ url.split('/')[-1]
        try:
            if 'tar.gz' in url:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(response.content)
                else:
                    print(response.status_code)

                ap = tarfile.open(file_name)

                for member in ap.getnames():
                    if '.jpg' in member:
                        ap.extract(path=abs_path+'/output_file/'+args.metabolite,member= member)
                if os.path.exists(abs_path+'/output_file/'+args.metabolite+'/'+ url.split('/')[-1].replace('.tar.gz','')):
                    for file in os.listdir(abs_path+'/output_file/'+args.metabolite+'/'+ url.split('/')[-1].replace('.tar.gz','')):
                        shutil.move(abs_path+'/output_file/'+args.metabolite+'/'+ url.split('/')[-1].replace('.tar.gz','')+'/'+file,abs_path+'/output_file/'+args.metabolite+'/0/'+url.split('/') [-1].replace('.tar.gz','')+'_'+file)
                    os.rmdir(abs_path+'/output_file/'+args.metabolite+'/'+ url.split('/')[-1].replace('.tar.gz',''))
                os.remove(file_name)
            elif '.pdf' in url:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(response.content)
                else:
                    print(response.status_code)

                pdf_file = fitz.open(file_name)
                save_index=0
                for page_index in range(len(pdf_file)):
                    page = pdf_file[page_index]
                    image_list = page.get_images()
                    for image_index, img in enumerate(page.get_images(), start=1):
                        xref = img[0]  
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image = Image.open(io.BytesIO(image_bytes))
                        image.save(abs_path+'/output_file/'+args.metabolite+'/0/'+url.split('/')[-1].replace('.pdf','')+'_'+str(save_index)+'.'+image_ext)
                        save_index+=1
                os.remove(file_name)

            else:
                response = requests.get(url, headers=headers, timeout=10)
                # Save the PDF
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(response.content)
                else:
                    print(response.status_code)
        except:
            pass
        
    return result_list
'''
if __name__ == '__main__':
    header= 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    args.metabolite= '2,3-butanediol'
    email= 'dlwnsrb@kaist.ac.kr'
    len_request=200
    bulkdownload(header, args.metabolite, email, len_request)'''




