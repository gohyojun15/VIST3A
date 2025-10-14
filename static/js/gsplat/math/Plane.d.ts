import { Vector3 } from './Vector3';
declare class Plane {
    readonly normal: Vector3;
    readonly point: Vector3;
    constructor(normal: Vector3, point: Vector3);
    intersect(origin: Vector3, direction: Vector3): Vector3 | null;
}
export { Plane };
