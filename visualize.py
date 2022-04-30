import pandas as pd
import numpy
import os, cv2

from typing import Union,List
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

label_format = '{:,.2%}'
df_original = pd.read_csv('scenario1_results_ham1000_resnet18_weight_trial01.csv')
df_lrp = pd.read_csv('scenario1_results_ham1000_resnet18_lrp_trial01.csv')

def load_image(image: Union[str, numpy.ndarray]) -> numpy.ndarray:
    # Image provided ad string, loading from file ..
    if isinstance(image, str):
        # Checking if the file exist
        if not os.path.isfile(image):
            print("File {} does not exist!".format(image))
            return None
        # Reading image as numpy matrix in gray scale (image, color_param)
        return cv2.imread(image)

    # Image alredy loaded
    elif isinstance(image, numpy.ndarray):
        return image

    # Format not recognized
    else:
        print("Unrecognized format: {}".format(type(image)))
        print("Unrecognized format: {}".format(image))
    return None

def show_images(images: List[numpy.ndarray]) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

    plt.show(block=True)


""" img_prototype_img_original11 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original11.png")
img_prototype_img_original_with_self_act11 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original_with_self_act11.png")
img_prototype_img11 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img11.png")


img_prototype_img_original22 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original22.png")
img_prototype_img_original_with_self_act22 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original_with_self_act22.png")
img_prototype_img22 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img22.png")

img_prototype_img_original30 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original30.png")
img_prototype_img_original_with_self_act30 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original_with_self_act30.png")
img_prototype_img30 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img30.png")

img_prototype_img_original48 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original48.png")
img_prototype_img_original_with_self_act48 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img-original_with_self_act48.png")
img_prototype_img48 = load_image("../ProtoPNet/saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/img/epoch-8/prototype-img48.png")

imglist=[
   img_prototype_img_original11,img_prototype_img_original_with_self_act11,img_prototype_img11,
     img_prototype_img_original22,img_prototype_img_original_with_self_act22,img_prototype_img22,
     img_prototype_img_original30,img_prototype_img_original_with_self_act30,img_prototype_img30,
     img_prototype_img_original48,img_prototype_img_original_with_self_act48,img_prototype_img48
    
]

cols = ["Validation Set Samples", "Aktivierungen (Prototypen)", "Geprunte Prototypen"]
rows = ["Sample 11","Sample 22","Sample 30","Sample 48"]

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 8),
sharex=True, sharey=True,
                           squeeze=True)

for ax, col in zip(axes[0], cols):
    ax.set_title(col, size='small')
    ax.set_aspect("equal", adjustable='box')

for i in range(4):
    axes[i][0].text(x = -20, y = 70, s = rows[i], rotation = 90)
    #axes[i][0].set_ylabel("ylabel")

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=0)
    ax.set_aspect("equal", adjustable='box')



axes[0,0].imshow(cv2.cvtColor(imglist[0], cv2.COLOR_BGR2RGB))
axes[0,1].imshow(cv2.cvtColor(imglist[1], cv2.COLOR_BGR2RGB))
axes[0,2].imshow(cv2.cvtColor(imglist[2], cv2.COLOR_BGR2RGB), aspect='equal',extent=(0, 222, 0, 222))
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')

axes[1,0].imshow(cv2.cvtColor(imglist[3], cv2.COLOR_BGR2RGB))
axes[1,1].imshow(cv2.cvtColor(imglist[4], cv2.COLOR_BGR2RGB))
axes[1,2].imshow(cv2.cvtColor(imglist[5], cv2.COLOR_BGR2RGB),aspect='equal',extent=(0, 222, 0, 222))
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')

axes[2,0].imshow(cv2.cvtColor(imglist[6], cv2.COLOR_BGR2RGB))
axes[2,1].imshow(cv2.cvtColor(imglist[7], cv2.COLOR_BGR2RGB))
axes[2,2].imshow(cv2.cvtColor(imglist[8], cv2.COLOR_BGR2RGB),aspect='equal', extent=(0, 222, 0, 222))
axes[2,0].axis('off')
axes[2,1].axis('off')
axes[2,2].axis('off')

axes[3,0].imshow(cv2.cvtColor(imglist[9], cv2.COLOR_BGR2RGB))
axes[3,1].imshow(cv2.cvtColor(imglist[10], cv2.COLOR_BGR2RGB))
axes[3,2].imshow(cv2.cvtColor(imglist[11], cv2.COLOR_BGR2RGB),aspect='equal', extent=(0, 222, 0, 222))
axes[3,0].axis('off')
axes[3,1].axis('off')
axes[3,2].axis('off')



plt.subplots_adjust(hspace=0.1, wspace=-0.5)

plt.savefig("pruned_rototypes2.png")
 """
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
data = {'Pruning-Kriterium': ['Aktivierungen (LRP)', 'Prototypen (ProtoPNet)', 'Magnitüden der Gewichte', 'Magnitüden der Gradienten'], 'Präzision nach \n dem Pruning': ['72,16%','81,32%', '72,16%', '-'] ,'Präzision nach \n anschließendem \n Fine-Tuning': ['84,04%', '90,11%', '84,04%','-']}   
df = pd.DataFrame(data)  
table = ax.table(cellText=df.values, cellLoc='center',colLabels=df.columns, colWidths=[0.3 for x in df.columns],loc='center',bbox=(0, 0, 1, 1))
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(2, 2)
for (row, col), cell in table.get_celld().items():
  if (row == 0) or (col == -1):
    cell.set_text_props(fontproperties=FontProperties(weight='bold', size=8))
fig.tight_layout()
plt.savefig("vergleich_pruningkriterien.png") 

print(df_original)
df_original['test_acc'] = df_original['test_acc']
df_lrp['test_acc'] = df_lrp['test_acc']

fig, ax = plt.subplots(figsize=(12, 8))
ax3 = ax.twinx()
df_original.plot(kind='line',x='step',y='flops',color=[(0.3,0.5,0.5)],ax=ax, label="FLOPs, Kriterium: Gewichte")
df_lrp.plot(kind='line',x='step',y='flops',color=[(0.3,0.7,0.7)],ax=ax,label="FLOPs, Kriterium: LRP")
df_original.plot(kind='line',x='step',y='test_acc',color=[(0.7,0.7,0.3)],ax=ax3,label="Test-Präzision, Kriterium: LRP")
df_lrp.plot(kind='line',x='step',y='test_acc',color=[(0.9,0.2,0.2)],ax=ax3,label="Test-Präzision, Kriterium: Gewichte")

ax.legend(bbox_to_anchor=(0.99,0.2), loc="lower right")
ax3.legend(bbox_to_anchor=(0.99,0.29), loc="lower right")
ticks_loc = ax3.get_yticks().tolist()
ax3.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
ax3.set_yticklabels([label_format.format(x) for x in ticks_loc])

ax3.set_ylabel("Test-Präzision (in Prozent)", labelpad=15)


ax.set_ylabel("FLOPs (in Billionen)", labelpad=15)
ax.set_xlabel("Pruning-Schritt")
plt.savefig("entwicklung_kriterien.png") 