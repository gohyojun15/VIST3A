import { Matrix3 } from './Matrix3';
import { Vector3 } from './Vector3';
declare class Quaternion {
    readonly x: number;
    readonly y: number;
    readonly z: number;
    readonly w: number;
    constructor(x?: number, y?: number, z?: number, w?: number);
    equals(q: Quaternion): boolean;
    normalize(): Quaternion;
    multiply(q: Quaternion): Quaternion;
    inverse(): Quaternion;
    apply(v: Vector3): Vector3;
    flat(): number[];
    clone(): Quaternion;
    static FromEuler(e: Vector3): Quaternion;
    toEuler(): Vector3;
    static FromMatrix3(matrix: Matrix3): Quaternion;
    static FromAxisAngle(axis: Vector3, angle: number): Quaternion;
    static LookRotation(direction: Vector3): Quaternion;
    toString(): string;
}
export { Quaternion };
