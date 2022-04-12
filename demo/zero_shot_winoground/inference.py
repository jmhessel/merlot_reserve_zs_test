"""
Demo for doing interesting things with a video
"""
import os
import sys
import json
import pickle
from tqdm import tqdm
sys.path.append('../')
import pprint
from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp
from finetune.common_data_utils import *
import tqdm

from datasets import load_dataset

try:
    dataset = load_dataset("facebook/winoground", use_auth_token=True)
except:
    print('downloaded images.')

image_root = os.path.expanduser('~/.cache/huggingface/datasets/downloads/extracted/f996d541cf936877d2ee83171a44d9736631ecb84fade4c88a1b0c78feef029d/images/')
n_images = len(os.listdir(image_root))
print('{} images'.format(n_images))

    
if not os.path.exists('zero_shot_winoground/examples.jsonl'):
    print('please download winoground examples jsonl and put them here.')

all_examples = []
with open('zero_shot_winoground/examples.jsonl') as f:
    for line in f:
        all_examples.append(json.loads(line.strip()))
print('{} examples'.format(len(all_examples)))
pprint.pprint(all_examples[0])

def load_image(path):
    image = Image.open(path).convert('RGB')
    image = resize_image(image, shorter_size_trg=450, longer_size_max=800)
    return image

    
# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
time_interval = 3.0
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)
#model = PretrainedMerlotReserve.from_pretrained(model_name='base', image_grid_size=grid_size)

text_accs, image_accs, group_accs = [], [], []

random_bl = False
# loop over each example..
for ex in tqdm.tqdm(all_examples):

    if not random_bl:
        cap0, cap1 = ex['caption_0'], ex['caption_1']
        im0, im1 = load_image(image_root+ex['image_0'] + '.png'), load_image(image_root+ex['image_1'] + '.png')
        segs = [{'frame': im0, 'text': '<|MASK|>'}, {'frame': im1, 'text': '<|MASK|>'}]
        video_pre = preprocess_video(segs, output_grid_size=grid_size)
        out_h = model.embed_video(**video_pre)
        out_h = out_h[video_pre['tokens'] == MASK]

        label_space = model.get_label_space([cap0, cap1])
        txt2im = jnp.einsum('bh,lh->bl', label_space, out_h)
    else:
        txt2im = np.random.random(size=(2, 2))

    # tiebreak randomly...
    s_C0_I0 = float(txt2im[0,0]) + (np.random.random() / 10**6)
    s_C1_I0 = float(txt2im[1,0]) + (np.random.random() / 10**6)
    s_C0_I1 = float(txt2im[0,1]) + (np.random.random() / 10**6)
    s_C1_I1 = float(txt2im[1,1]) + (np.random.random() / 10**6)

    print(s_C0_I0, s_C1_I0, s_C0_I1, s_C1_I1)
    if s_C0_I0 > s_C1_I0 and s_C1_I1 > s_C0_I1:
        text_accs.append(1.0)
    else:
        text_accs.append(0.0)

    if s_C0_I0 > s_C0_I1 and s_C1_I1 > s_C1_I0:
        image_accs.append(1.0)
    else:
        image_accs.append(0.0)

    if text_accs[-1] == 1.0 and image_accs[-1] == 1.0:
        group_accs.append(1.0)
    else:
        group_accs.append(0.0)

        
print('text acc: {:.2f}, image acc: {:.2f}, group acc: {:.2f}'.format(
    100*np.mean(text_accs),
    100*np.mean(image_accs),
    100*np.mean(group_accs)))

