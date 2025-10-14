declare class SplatvData {
    static RowLength: number;
    private _vertexCount;
    private _positions;
    private _data;
    private _width;
    private _height;
    serialize: () => Uint8Array;
    constructor(vertexCount: number, positions: Float32Array, data: Uint32Array, width: number, height: number);
    static Deserialize(data: Uint8Array, width: number, height: number): SplatvData;
    get vertexCount(): number;
    get positions(): Float32Array<ArrayBufferLike>;
    get data(): Uint32Array<ArrayBufferLike>;
    get width(): number;
    get height(): number;
}
export { SplatvData };
