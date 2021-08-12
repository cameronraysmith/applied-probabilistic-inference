---
jupyter:
  celltoolbar: Slideshow
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md,py:percent
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.6
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
---

# Plotting setup

```python
import matplotlib.pyplot as plt
```

## Show available fonts

```python
# http://jonathansoma.com/lede/data-studio/matplotlib/list-all-fonts-available-in-matplotlib-plus-samples/
# List all fonts available in matplotlib plus samples

import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))
```

```python jupyter={"outputs_hidden": true}
# ?matplotlib.font_manager.fontManager.addfont
```

## Add latin modern fonts

```python
# https://www.archlinux.org/packages/community/any/otf-latin-modern/
# !sudo pacman -Syu --needed --noconfirm otf-latin-modern inkscape
# !brew cask install font-latin-modern
# !apt-get install -y fonts-lmodern
# fonts_path_ubuntu = "/usr/share/texmf/fonts/opentype/public/lm/"
# fonts_path_macos = "~/Library/Fonts/"
fonts_path_arch = "/usr/share/fonts/OTF/"
matplotlib.font_manager.fontManager.addfont(fonts_path_arch + "lmsans10-regular.otf")
matplotlib.font_manager.fontManager.addfont(fonts_path_arch + "lmroman10-regular.otf")
```

## Set matplotlib to use Latin Modern fonts

```python
from IPython.display import set_matplotlib_formats
#%matplotlib inline
set_matplotlib_formats('svg') # use SVG backend to maintain vectorization
plt.style.use('default') #reset default parameters
# https://stackoverflow.com/a/3900167/446907
plt.rcParams.update({'font.size': 16,
                     'font.family': ['sans-serif'],
                     'font.serif': ['Latin Modern Roman'] + plt.rcParams['font.serif'],
                     'font.sans-serif': ['Latin Modern Sans'] + plt.rcParams['font.sans-serif']})
```

## Check rcParams after update and list fonts available to matplotlib

```python
# plt.rcParams.values

# code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

# HTML("<div style='column-count: 2;'>{}</div>".format(code))
```

## Create a test plot

```python
# import numpy as np

# t = np.arange(-1.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)

# fig, ax = plt.subplots()
# ax.plot(t, s)

# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()

# plt.savefig("fig/test-plot.svg", bbox_inches="tight");
# !inkscape fig/test-plot.svg --export-filename=fig/test-plot.pdf;
```
