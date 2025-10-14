import { Quaternion } from './Quaternion';
import { Vector3 } from './Vector3';
declare class Matrix3 {
    readonly buffer: number[];
    constructor(n11?: number, n12?: number, n13?: number, n21?: number, n22?: number, n23?: number, n31?: number, n32?: number, n33?: number);
    equals(m: Matrix3): boolean;
    multiply(v: Matrix3): Matrix3;
    clone(): Matrix3;
    static Eye(v?: number): Matrix3;
    static Diagonal(v: Vector3): Matrix3;
    static RotationFromQuaternion(q: Quaternion): Matrix3;
    static RotationFromEuler(m: Vector3): Matrix3;
    toString(): string;
}
export { Matrix3 };
