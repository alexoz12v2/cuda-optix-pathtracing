# BVH handling

We can distinguish 3 separate phases to implement a Bounding Volume Hierarchy

- Step By Step Construction
    - produces a _binary_ BVH output, whose format is well known and common to all architectures, hence should be 
     **pointerless**
- Layout conversion
    - In the CPU path, we convert the pointerless binary representation into a multi-branch BVH
- Traversal
    - Specialized function to intersect a ray
