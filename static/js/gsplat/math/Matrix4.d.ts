import { Quaternion } from './Quaternion';
import { Vector3 } from './Vector3';
declare class Matrix4 {
    readonly buffer: number[];
    constructor(n11?: number, n12?: number, n13?: number, n14?: number, n21?: number, n22?: number, n23?: number, n24?: number, n31?: number, n32?: number, n33?: number, n34?: number, n41?: number, n42?: number, n43?: number, n44?: number);
    equals(m: Matrix4): boolean;
    multiply(m: Matrix4): Matrix4;
    clone(): Matrix4;
    determinant(): number;
    invert(): Matrix4;
    static Compose(position: Vector3, rotation: Quaternion, scale: Vector3): Matrix4;
    toString(): string;
}
export { Matrix4 };
