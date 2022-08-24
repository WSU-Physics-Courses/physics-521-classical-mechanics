---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (phys-521-2022)
  language: python
  name: phys-521-2022
---

# Manim

+++

[Manim] is a Python library for creating mathematical animations in the style of those used by the [3Blue1Brown] [YouYube channel](https://www.youtube.com/3blue1brown).

+++

## Examples

Here are some examples from the [First Steps with Manim] notebook.

```{code-cell} ipython3
from manim import *

config.media_width = "60%"
```

```{code-cell} ipython3
%%manim -v WARNING -qm CircleToSquare

class CircleToSquare(Scene):
    def construct(self):
        blue_circle = Circle(color=BLUE, fill_opacity=0.5)
        green_square = Square(color=GREEN, fill_opacity=0.8)
        self.play(Create(blue_circle))
        self.wait()
        
        self.play(Transform(blue_circle, green_square))
        self.wait()
```

## Manim and JupyterBook

+++

Unfortunately, the videos generated above will not work with JupyterBook output.  

### What Works

Static Images
: This works out of the box, just use `--format=gif` or `--format=png`.  Don't try to do
  animations though (animated gifs).

Embeded Movies
: If you can find where the rendered movie is stored, then you can import it with
  {py:class}`IPython.display.Video` class with the `embed=True` option.

Movies in `_static`
: If you are careful to put your movies in the `_static/` directory that Sphinx copies
  (see [`html_static_path`]), then you can embed them with the `<video>` tag:

  ```html
  <video autoplay loop width='420' 
         src='../_static/CircleToSquare.mp4' type='video/mp4'/>
  ```
  
  gives
  
  <video autoplay loop width='420' src='../_static/CircleToSquare.mp4' type='video/mp4'>
  </video>
  
  **Caveats:**
  * You need to know where you are.  The relative path `../_static` above is used
    because this document is nested one level down.  This makes your document somewhat
    brittle and you cannot move it to a different level.
    
    I played with using an absolute path `/_static`, but this breaks [RTD] for example
    which translates to
    
    * `https://...readthedocs.io/_static/CircleToSquare.mp4`
    
    but should be

    * `https://...readthedocs.io/en/latest/_static/CircleToSquare.mp4`
    
    
  * This may work if you run the notebook with Jupyter, but can fail.  For example, if
    you run jupyter in the same directory as this notebook, then `../` will be above the
    `tree` and it will fail.

The `manim` Sphinx directive
: You can include the {mod}`manim.utils.docbuild.manim_directive` directive in your
  `conf.py` file, then use Sphinx.  Unfortunately, you must fallback to the [`eval-rst`
  directive] which is a bit of a pain (code must be indented)

+++

#### Static Images

Static images work, so we can use [Manim] as a drawing tool, but we cannot "play" animations out of the box.

```{margin}
The first line contains the `%%manim` cell magic ({py:meth}`manim.utils.ipython_magic.ManimMagic.manim`) which basically allows you to pass command-line arguments to `manim`.  The non-obvious options we use here are:

`-v`: Verbosity

`-r`: Resolution

`-qm`: Medium render quality

`--format`: One of `[png|gif|mp4|webm|mov]`

The final argument is the name of the input file - the {py:class}`manim.scene.scene.Scene` subclass here.
```

```{code-cell} ipython3
%%manim -v WARNING --progress_bar None -r 400,200 --format=gif -qm CircleAndSquare

class CircleAndSquare(Scene):
    def construct(self):
        blue_circle = Circle(color=BLUE, fill_opacity=0.5)
        green_square = Square(color=GREEN, fill_opacity=0.8)
        green_square.shift(2*RIGHT)
        self.add(blue_circle)
        self.add(green_square)
```

#### Embedded Videos

+++

Embedding videos works, but you need to know where the files are:

```{code-cell} ipython3
from IPython.display import Video
Video('./media/videos/OtherNotes/720p30/CircleToSquare.mp4', 
      width=500, 
      embed=True,
      html_attributes="controls muted loop autoplay")
```

The following will work with Jupyter notebooks since the path is relative to the notebook, but will not work with Sphinx because these movies are not copied to the output directory:

```html
<video autoplay loop width='420' 
       src='./media/videos/OtherNotes/720p30/CircleToSquare.mp4' type='video/mp4'>
</video>
```
<video autoplay loop width='420' src='./media/videos/OtherNotes/720p30/CircleToSquare.mp4' type='video/mp4'>
</video>

+++

#### Movies in `_static`

+++

If you know where you are, you can put the movie in the appropriate `_static` folder that is copied over by Sphinx.  Then you can embed them:

```html
<video autoplay loop width='420' 
       src='../_static/CircleToSquare.mp4' type='video/mp4'>
</video>
```
<video autoplay loop width='420' src='../_static/CircleToSquare.mp4' type='video/mp4'>
</video>

```{code-cell} ipython3
%%manim -v WARNING --disable_caching --progress_bar None -qm -o ../../../../_static/CircleToSquare.mp4 CircleToSquare

class CircleToSquare(Scene):
    def construct(self):
        blue_circle = Circle(color=BLUE, fill_opacity=0.5)
        green_square = Square(color=GREEN, fill_opacity=0.8)
        self.play(Create(blue_circle))
        self.wait()
        
        self.play(Transform(blue_circle, green_square))
        self.wait()
```

This can be displayed in python:

```{code-cell} ipython3
from IPython.display import Video
Video('../_static/CircleToSquare.mp4', 
      width=500, 
      embed=False,
      mimetype="video/mp4",
      html_attributes="controls muted loop autoplay")
```

Looking under the hood, what happens here is that the following files are generated relative to this notebook.  This is why we need such and ugly output path for `%%manim`.

```
`-- media
    |-- images
    |   `-- OtherNotes
    |       `-- CircleAndSquare_ManimCE_v0.13.1.png
    |-- jupyter
    |   |-- CircleAndSquare@2022-01-08@00-13-03.png
    |   |-- CircleToSquare@2022-01-08@00-13-03.mp4
    |   `-- CircleToSquare@2022-01-08@00-13-04.gif
    `-- videos
        `-- OtherNotes
            |-- 200p30
            |   |-- CircleToSquare_ManimCE_v0.13.1.gif
            |   `-- partial_movie_files
            |       |-- CircleAndSquare
            |       `-- CircleToSquare
            |           |-- partial_movie_file_list.txt
            |           |-- uncached_00000.mp4
            |           |-- uncached_00001.mp4
            |           |-- uncached_00002.mp4
            |           `-- uncached_00003.mp4
            `-- 720p30
                |-- CircleToSquare.mp4
                `-- partial_movie_files
                    `-- CircleToSquare
                        |-- 1342159004_1441543657_2959157631.mp4
                        |-- 1353618911_2262467609_2655279732.mp4
                        |-- 1353618911_398514950_1253908064.mp4
                        |-- 1353618911_398514950_694114964.mp4
                        `-- partial_movie_file_list.txt

```

```{code-cell} ipython3
%%manim -v WARNING --progress_bar None --disable_caching -qm BannerExample

config.media_width = "75%"

class BannerExample(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        banner_large = ManimBanner(dark_theme=False).scale(0.7)
        self.play(banner_large.create())
        self.play(banner_large.expand())
```

#### Manim Sphinx Directive

The {mod}`manim.utils.docbuild.manim_directive` directive works:

```python
# conf.py
...
extensions = [
    "manim.utils.docbuild.manim_directive",
    ...
]
...
```

It needs to be escaped using the [`eval-rst` directive]

````markdown
```{eval-rst}
.. manim:: MyScene

   class MyScene(Scene):
       def construct(self):
           self.camera.background_color = "#ece6e2"
           banner_large = ManimBanner(dark_theme=False).scale(0.7)
           self.play(banner_large.create())
           self.play(banner_large.expand())
```
````

gives

```{eval-rst}
.. manim:: MyScene

   class MyScene(Scene):
       def construct(self):
           self.camera.background_color = "#ece6e2"
           banner_large = ManimBanner(dark_theme=False).scale(0.7)
           self.play(banner_large.create())
           self.play(banner_large.expand())
```

`````{note}

The following gives a `WARNING: Directive 'manim' cannot be mocked: MockingError: 
MockStateMachine has not yet implemented attribute 'insert_input'` message and fails to
render.  This is probably also why the previous directive does not give a proper width.

````
```{manim} MyScene

class MyScene(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        banner_large = ManimBanner(dark_theme=False).scale(0.7)
        self.play(banner_large.create())
        self.play(banner_large.expand())
```
````
`````


## References

* [`manim` command-line arguments (CLI)](https://docs.manim.community/en/stable/tutorials/configuration.html#command-line-arguments)

[`eval-rst` directive]: <https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html#syntax-directives-parsing>
[Manim]: <https://www.manim.community/>
[3Blue1Brown]: <https://www.3blue1brown.com/>
[`html_static_path`]: <https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_static_path>
[First Steps with Manim]: <https://hub.gke2.mybinder.org/user/manimcommunity-jupyter_examples-heewn38p/notebooks/First%20Steps%20with%20Manim.ipynb>
