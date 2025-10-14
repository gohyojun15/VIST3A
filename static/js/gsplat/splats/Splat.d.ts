import { SplatData } from './SplatData';
import { Object3D } from '../core/Object3D';
import { Matrix4 } from '../math/Matrix4';
import { Box3 } from '../math/Box3';
declare class Splat extends Object3D {
    selectedChanged: boolean;
    colorTransformChanged: boolean;
    private _data;
    private _selected;
    private _colorTransforms;
    private _colorTransformsMap;
    private _bounds;
    recalculateBounds: () => void;
    constructor(splat?: SplatData | undefined);
    saveToFile(name?: string | null, format?: "splat" | "ply"): void;
    get data(): SplatData;
    get selected(): boolean;
    set selected(selected: boolean);
    get colorTransforms(): Matrix4[];
    get colorTransformsMap(): Map<number, number>;
    get bounds(): Box3;
    clone(): Splat;
}
export { Splat };
