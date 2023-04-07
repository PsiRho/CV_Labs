
# CV_Labs

---

###### Collection of code that's been used for the IDATG2206 Computer Vision course at NTNU.

---

## Disclaimer

This code should probably not be used for anything where you need it to work!

It's single use. Don't be surprised if something brakes after using it one time. If you as much as think about changing
a single character it will probably break. There is no proper testing and there is no error handling. I take no
responsibility for anything related to the code.

Other than that, feel free to use the code.

---

## Example images

<figure>
    <img src="output_img/included_in_report/images/medianFilter_LoG_kernelSize5_computedSigma_threshold150_sigmaRatio1.6.png" width="500"
         alt="Flower">
    <figcaption>Edge detection with laplacian of gaussian (LoG) filter.</figcaption>
</figure>

<figure>
    <img src="output_img/LoG_ksz=9_color.jpg" width="500"
         alt="Flower colored">
    <figcaption>Edge detection with laplacian of gaussian (LoG) filter colored.</figcaption>
</figure>

<figure>
    <img src="output_img/LoG_ksz=9_color_inv.jpg" width="500"
         alt="Flower inverted and inverted">
    <figcaption>Same as above with inverted thresholding.</figcaption>
</figure>

<figure>
    <img src="output_img/Tiger_LoG_ksz=9_color_inv.jpg" width="500"
         alt="Tiger">
    <figcaption>Same as above with different image.</figcaption>
</figure>


---

## Known issues

You might get a warning when running the code in PyCharm.

```
-------------------------------------------------------------------------------
pydev debugger: CRITICAL WARNING: This version of python seems to be incorrectly compiled (internal generated filenames are not absolute)
pydev debugger: The debugger may still function, but it will work slower and may miss breakpoints.
pydev debugger: Related bug: https://bugs.python.org/issue1666807

-------------------------------------------------------------------------------
```

The issue in the link above is migrated to https://github.com/python/cpython/issues/44604 where a fix is proposed.

A quick fix is to add the following to the run configuration in the interpreter options in PyCharm's Python Console
preferences (Preferences > Build, Execution, Deployment > Console > Python Console):

```
-Xfrozen_modules=off
```

---