import { Vector3 } from './Vector3';
declare class Box3 {
    min: Vector3;
    max: Vector3;
    constructor(min: Vector3, max: Vector3);
    contains(point: Vector3): boolean;
    intersects(box: Box3): boolean;
    size(): Vector3;
    center(): Vector3;
    expand(point: Vector3): void;
    permute(): void;
}
export { Box3 };
