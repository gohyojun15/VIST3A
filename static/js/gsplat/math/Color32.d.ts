declare class Color32 {
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
    constructor(r?: number, g?: number, b?: number, a?: number);
    flat(): number[];
    flatNorm(): number[];
    toHexString(): string;
    toString(): string;
}
export { Color32 };
