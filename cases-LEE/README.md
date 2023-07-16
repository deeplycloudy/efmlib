# Calibration notes

According to the triboelectric series, hair charges positively, and foam negatively.

Define the electric field polarity using the physics sign convention: field points in the direction of motion of an imagined + charge in the field.

Therefore, , when the calibration foam is held _ _ _ relative to the efm, its charge produces _ _ _.
- above: +Ez
- below: -Ez
- north: +Ey
- east: +Ex
Ez will always be larger than Ex and Ey since the foam can be held very close to the spheres, both above and below. 

The north and east components can only be determined when the instrument is rotating.

Charge above followed by charge below is the usual sequence followed during checkout.

These are the basic physical principles used to understand the calibration data.

## Polarity for individual instruments

Early in the campaign, for checkouts on or before 21 Nov and the flights during IOP2, the Sleet and Graupel instruments had their analog boards mounted in the reversed direction in the spheres. Sleet's board orientation was corrected during the checkout on 19 Dec, before its next use during IOP10. Graupel was never reused after its first flight. Therefore, the board orientation only needs to be corrected for the IOP2 flights of Graupel and Sleet.

Each IMU model has its own coordinate convention, which is the {x, y, z} recorded in the raw EFM data. The boards used during LEE had the LSM9DS1 IMU had the chips mounted on their circuit board as described in the [NSSL EFM hardware/firmware repository on GitHub](https://github.com/LeemanGeophysicalLLC/NSSL_EFM/issues/71). Specifically, when looking at the board with the "Rev 3.1" label at the bottom, the z axis points out of the board (toward the viewer) for all three channels. -y for all channels is toward the top of the board. The convention for x is right-handed for the magentic field sensor, and left-handed for the accelerometer and gyroscope.

The y accelerometer should be in phase with spin. If the board was mounted upside down, the centrifugal offset in the y channel will also be in the opposite direction, and the sign for Ez will flip.

If the board was mounted on the opposite side of the post, but in the same up-down orientation, the sign conventions for x and z will change.
TODO: diagnose if any boards were mounted opposite.

Comparing the cal data from IOP1-Ice and IOP2-Graupel, we observe opposite signs in the centrifugal offset in the y-channel (positive for Ice and negative for Graupel), confirming we also need to flip the sign of E relative to one another. Apparently "reversed" also means "upside down".
