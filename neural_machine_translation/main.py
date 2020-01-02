import sys
import json
import numpy as np
import torch
from torch.utils import data

# = = = = = = = = = = =

path_root = '/NMT'

path_to_model = path_root + '/code/'
sys.path.insert(0, path_to_model)

from model import seq2seqModel

path_to_data = path_root + 'data/'
path_to_save_models = path_root + 'models/'

# = = = = = = = = = = =

class Dataset(data.Dataset):
  def __init__(self, pairs):
        self.pairs = pairs

  def __len__(self):
        return len(self.pairs) # total nb of observations

  def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)


def load_pairs(train_or_test):
    with open(path_to_data + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()
    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]
    return pairs_tmp


# = = = = = = = = = = =

do_att = True # should always be set to True
is_prod = True # production mode or not

if not is_prod:
        
    pairs_train = load_pairs('train')
    pairs_test = load_pairs('test')
    
    with open(path_to_data + 'vocab_source.json','r') as file:
        vocab_source = json.load(file) # word -> index
    
    with open(path_to_data + 'vocab_target.json','r') as file:
        vocab_target = json.load(file) # word -> index
    
    vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word
    
    print('data loaded')
        
    training_set = Dataset(pairs_train)
    test_set = Dataset(pairs_test)
    
    print('data prepared')
    
    print('= = = attention-based model?:',str(do_att),'= = =')
    
    model = seq2seqModel(vocab_s=vocab_source,
                         source_language='english',
                         vocab_t_inv=vocab_target_inv,
                         embedding_dim_s=40,
                         embedding_dim_t=40,
                         hidden_dim_s=30,
                         hidden_dim_t=30,
                         hidden_dim_att=20,
                         do_att=do_att,
                         padding_token=0,
                         oov_token=1,
                         sos_token=2,
                         eos_token=3,
                         max_size=30) # max size of generated sentence in prediction mode
    
    model.fit(training_set,test_set,lr=0.001,batch_size=64,n_epochs=20,patience=2)
    model.save(path_to_save_models + 'my_model.pt')

else:
    
    model = seq2seqModel.load(path_to_save_models + 'pretrained_moodle.pt')
    
    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
                'This river is full of fish.', # plein vs pleine (accord)
                'The fridge is full of food.',
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace']
    
    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt,model.predict(elt)))

#######################################################################
							# TO MAKE PLOTS
#######################################################################
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return text

## example of a plot

model = seq2seqModel.load(path_to_save_models + 'pretrained_moodle.pt')
sentence = 'The fridge is full of food.'
translation = model.predict(sentence)

ph_en = sentence.split()
ph_fr = translation.split()

alphas = model.alphas
n = min(len(ph1_en),len(ph1_fr))
coeff = torch.stack(alphas, dim=1).squeeze(2)[:n,:n]
coeff = np.array(coeff.detach().cpu().clone().numpy())
print(coeff.shape)

fig, ax = plt.subplots()
fig.set_size_inches(10, 7, forward=True)

im, cbar = heatmap(coeff, ph1_en, ph1_fr, ax=ax, cmap="gist_gray", cbarlabel="source/target alignments")
texts = annotate_heatmap(im, valfmt="{x:.4f}")

fig.tight_layout()
plt.show()

