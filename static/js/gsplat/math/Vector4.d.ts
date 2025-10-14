import { Matrix4 } from './Matrix4';
declare class Vector4 {
    readonly x: number;
    readonly y: number;
    readonly z: number;
    readonly w: number;
    constructor(x?: number, y?: number, z?: number, w?: number);
    equals(v: Vector4): boolean;
    add(v: number): Vector4;
    add(v: Vector4): Vector4;
    subtract(v: number): Vector4;
    subtract(v: Vector4): Vector4;
    multiply(v: number): Vector4;
    multiply(v: Vector4): Vector4;
    multiply(v: Matrix4): Vector4;
    dot(v: Vector4): number;
    lerp(v: Vector4, t: number): Vector4;
    magnitude(): number;
    distanceTo(v: Vector4): number;
    normalize(): Vector4;
    flat(): number[];
    clone(): Vector4;
    toString(): string;
}
export { Vector4 };
