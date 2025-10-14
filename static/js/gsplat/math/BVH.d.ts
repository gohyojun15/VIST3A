import { Box3 } from './Box3';
declare class BVHNode {
    bounds: Box3;
    boxes: Box3[];
    left: BVHNode | null;
    right: BVHNode | null;
    pointIndices: number[];
    constructor(bounds: Box3, boxes: Box3[], pointIndices: number[]);
    private split;
    queryRange(range: Box3): number[];
}
declare class BVH {
    root: BVHNode;
    constructor(bounds: Box3, boxes: Box3[]);
    queryRange(range: Box3): number[];
}
export { BVH };
