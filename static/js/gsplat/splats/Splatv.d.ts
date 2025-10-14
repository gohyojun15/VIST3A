import { Object3D } from '../core/Object3D';
import { SplatvData } from './SplatvData';
declare class Splatv extends Object3D {
    private _data;
    constructor(splat: SplatvData);
    get data(): SplatvData;
}
export { Splatv };
