import { CameraData } from './CameraData';
import { Object3D } from '../core/Object3D';
import { Vector3 } from '../math/Vector3';
declare class Camera extends Object3D {
    private _data;
    screenPointToRay: (x: number, y: number) => Vector3;
    constructor(camera?: CameraData | undefined);
    get data(): CameraData;
}
export { Camera };
