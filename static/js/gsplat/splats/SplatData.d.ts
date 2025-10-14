import { Vector3 } from '../math/Vector3';
import { Quaternion } from '../math/Quaternion';
declare class SplatData {
    static RowLength: number;
    changed: boolean;
    detached: boolean;
    private _vertexCount;
    private _positions;
    private _rotations;
    private _scales;
    private _colors;
    private _selection;
    translate: (translation: Vector3) => void;
    rotate: (rotation: Quaternion) => void;
    scale: (scale: Vector3) => void;
    serialize: () => Uint8Array;
    reattach: (positions: ArrayBufferLike, rotations: ArrayBufferLike, scales: ArrayBufferLike, colors: ArrayBufferLike, selection: ArrayBufferLike) => void;
    constructor(vertexCount?: number, positions?: Float32Array | null, rotations?: Float32Array | null, scales?: Float32Array | null, colors?: Uint8Array | null);
    static Deserialize(data: Uint8Array): SplatData;
    get vertexCount(): number;
    get positions(): Float32Array<ArrayBufferLike>;
    get rotations(): Float32Array<ArrayBufferLike>;
    get scales(): Float32Array<ArrayBufferLike>;
    get colors(): Uint8Array<ArrayBufferLike>;
    get selection(): Uint8Array<ArrayBufferLike>;
    clone(): SplatData;
}
export { SplatData };
