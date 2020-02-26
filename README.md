# WaveBlocks
A polimorfic optimization framework for fluorescent microscopes based on Pytorch, capable of:
* Simulation.
* Calibration.
* PSF engineering.
* Joint optimization of optics and reconstruction, segmentation and arbitrary optimization tasks.
* etc.

Wave-optics involves complex diffraction integrals that when stacking many optical elements become very hard to derivate and optimize. Here we take advantage of the linearity of the posible operations and modularize them into blocks, enabling building arbitrarly large systems.

By building a WaveBlocks microscope similar to the one in your lab you can first calibrate the unknown parameters, like an accurate PSF, distance between elements, and even errors caused by aberrations and SLM diffraction. Once with the capability of simulating images similar to your microscope you can plug in any Pytorch based network to the imaging system and find and optimize the optimal optical parameters together with your network.

## Workflow
* Each optical element in a microscope (lenses, propagation through a medium, cameras, PSFs, Spatial Light modulator, appertures, Micro-lens arrays, etc) is represented by a block (Lego like) that can be asembled in any order, each block has as an input/output a complex Wave-Front and processes the input given the wave-optics behavior of the block.

* Most of the variables of the blocks are optimizable, allowing straight forward tasks like calibrating distances between optical elements, correction factors, compute real PSFs, all based on real data. 

* All the blocks are based on a OpticBlock class. Which takes care of gathering the parameters to optimize selected by the user and return them to be fed to the optimizer (Adam, SGD, etc).

## Posible blocks
<dl><dt>OpticConfig (nn.Module)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>wavelength</li>
            <li>wave_number</li>
            <li>medium_refractive_index</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>OpticalBlock (nn.Module)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>list:members_to_learn<\li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>WavePropagation (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>sampling_rate</li>
            <li>distance</li>
            <li>field_length<\li>
            <li>method = 'Rayleight-Sommerfield'</li>
            <li>propagation_function<\li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>find_method</li>
            <li>propagate</li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            <li>find_method function finds the correct diffraction approximation (Rayleight-sommerfield, Fresnel or Fraunhofer) for the given distance and field size(apperture), and populates propagation_function</li>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>Lens (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>focal_length<\li>
            <li>apperture_length<\li>
            <li>WavePropagation(?)</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>focus_to_focus<\li>
            <li>image_to_object<\li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            <li>This class can propagate a wave from the focal plane to the back focal plane or from any point i in front of the lens, to any point o in the back of the lens.<\li>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>DiffractiveElement (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>apperture_length</li>
            <li>sampling_rate</li>
            <li>phase_function</li>
            <li>function_image</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>apply_pointwise<\li>
            <li>apply_convolution<\li>
            <li>compute_function_image<\li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            <li>This class can be used for any apperture, mask, phase mask or spatial light modulator (SLM).</li>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>MicroLensArray (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>focal_lenght</li>
            <li>pixel_size</li>
            <li>image_shape</li>
            <li>trans_function (transmitance function of single block)</li>
            <li>block_image (transmitance image of single block)</li>
            <li>function_image (block_image replicated accross image)</li>
            <li>block_shape (Nnum)</li>
            <li>block_separation (pixel separation between blocks)</li>
            <li>block_offset_odd_row (for hexagonal grids)</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>compute_block_image<\li>
            <li>compute_full_image<\li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            <li>Usefull for periodic structures like micro lens arrays (MLA)</li>
            <li>apply_convolution returns a 4D psf as when convolving the input PSF at the different positions of the micro-lens create a different 2D pattern.</li>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>Camera (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>camera_size</li>
            <li>pixel_size</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>compute_intensity<\li>
            <li>apply_space_invariant_psf</li>
            <li>apply_space_variant_psf</li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            <li>Intensity = abs(complex_waveform).^2</li>
            </ul>
        </li>
    </ul>
</dl>

<dl><dt>PSF (OpticalBlock)</dt>
    <ul>
        <li>
            <dt>Members:</dt>
            <ul>
            <li>OpticConfig</li>
            <li>psf_size</li>
            <li>voxel_size</li>
            <li>psf_stack</li>
            </ul>
        </li>
        <li>
            <dt>Methods:</dt>
            <ul>
            <li>compute_psf</li>
            </ul>
        </li>
        <li>
            <dt>Notes:</dt>
            <ul>
            </ul>
        </li>
    </ul>
</dl>

# Asumptions
The following design assumptions and functionalites are taken into account:
* All the blocks work with a wave-optics approach of light simulation, appropiate for difracted limited systems like microscopes.
* Light sources are assumed to be incoherent (as in fluorescence microscopy) (todo-future: also coherent for lasers and holography).
* The scalar diffraction theory is assumed (paraxial approximation, small angles)
