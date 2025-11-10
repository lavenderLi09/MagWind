# Solar Corona Magnetic Field with Solar Wind (MagWind)
## Authors and Contributors
### Lead Author
- **Yihua Li**  
  *Nanjing University, School of Astronomy and Space Science*  
  GitHub: [@lavenderLi09](https://github.com/lavenderLi09)  
  Email: yihuali@smail.nju.edu.cn
  
### Co-authors
- **Guoyin Chen**  
  *Nanjing University, School of Astronomy and Space Science / Rosseland Centre for Solar Physics, University of Oslo*  
  GitHub: [@gychen-NJU]([https://github.com/gychen-NJU])

We thank all the contributors for their valuable input and support.

## Purpose

## Methods
### PFSS (Potential Field Source Surface) Model

The Potential-field Source-surface (PFSS) model offers a straightforward way to represent the global coronal magnetic field. 
It assumes that electric currents in the corona have minimal impact on the large-scale field structure, therefore, the over-all magnetic field
can be simplied to a potential field. We compute the PFSS model using spherical harmonics with the lower-boundary data from SDO/HMI synoptic maps.
The code can perform parallel computations on both CPU and GPU.

### SCS (Schatten Current Sheet) Model

The Schatten Current Sheet (SCS) model is an empirical magnetostatic model that describes the large-scale coronal and heliospheric magnetic field 
structure, which is proposed in ([https://ui.adsabs.harvard.edu/abs/1969SoPh....6..442S/abstract]). The model assumes that 
the coronal magnetic field can be approximated by a potential field in the solar corona, 
which transitions to a radial (current-free) configuration beyond a source surface, typically placed at 2.5 solar radii.

Key features:
- **Source Surface**: Assumes the magnetic field becomes radial at a specified radius
- **Current Sheet**: Describes the heliospheric current sheet separating opposite magnetic polarities
- **Applications**: Solar wind modeling, heliospheric field structure, space weather prediction

### OFF (OutFlow Field) Model

Based on Rice & Yeates, 2021 ([https://iopscience.iop.org/article/10.3847/1538-4357/ac2c71]), the outflow field is a global solar magnetic field model
that obtained from a magneto-frictional equbilirium, which contains a Parker solar wind profile in the magneto-frictional equation. Compared to the PFSS model, 
it provides different open fluxes by adjusting on the solar wind velocity, thereby allowing flexible adaptation based on different solar activity and specialties 
of events.

