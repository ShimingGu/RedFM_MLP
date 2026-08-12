# Target image centring audit

## Current conclusion

A cutout QA script should be completed before rebuilding the AION image embeddings. However, the current cutouts are not obviously miscentred.

An audit of 200 randomly selected retained objects found median positive-flux centroid offsets of approximately 0.8--1.2 pixels across grizy. The 90th-percentile offsets were approximately 4--7 pixels, but a centroid formed from every positive pixel is sensitive to blends, neighbouring sources, noise, and masked pixels. These larger offsets do not by themselves demonstrate a WCS-centring failure.

The sampled cutouts were also consistent with the HSC coadd photometric zeropoint of 27 expected by the AION HSC image codec. The immediate priority is therefore to distinguish genuine target displacement from contamination and masking, rather than blindly shifting every image.

## QA script requirements

The script should:

- Display the five individual grizy cutouts, an RGB composite, and the corresponding validity or mask maps.
- Overlay both the catalogue/WCS centre and a robust measured target centroid.
- Estimate the target centroid primarily from the i-band within a small aperture around the catalogue position, rather than using all positive pixels in the 96 x 96 cutout.
- Apply a single shared astrometric shift to every band if recentering is required. Bands must not be centred independently because that would destroy their relative alignment and colour information.
- Limit any automatic correction to a conservative displacement, initially about four pixels. Larger inferred shifts should be flagged for inspection or rejection.
- Generate contact sheets for both a random sample and the worst-offset tail.
- Report per-band cutout coverage, central signal-to-noise, masked fraction, centroid displacement, neighbour-contamination indicators, and aperture-flux consistency with the catalogue photometry.
- Preserve the original object ID, sky coordinates, assigned tile, pixel coordinates, proposed shift, and QA decision in a machine-readable table.

## Recommended workflow

1. Run the script in audit-only mode without modifying cutouts.
2. Inspect random objects and the largest-offset tail to determine whether offsets are caused by WCS errors, blends, masks, or low signal-to-noise.
3. Define conservative acceptance, recentering, and rejection criteria from those results.
4. Re-extract cutouts only if a genuine centring problem is demonstrated.
5. Cache the complete AION encoder-token sequences for the accepted objects and train the paper-style learned attentive pooling head.

Blind recentering is unsafe in crowded or blended regions because it can move a cutout from the catalogue target onto a brighter neighbour. The QA script should therefore flag uncertain objects instead of forcing every cutout to a flux maximum.
