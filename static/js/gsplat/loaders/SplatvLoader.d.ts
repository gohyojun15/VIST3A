import { Camera } from '../cameras/Camera';
import { Scene } from '../core/Scene';
import { Splatv } from '../splats/Splatv';
declare class SplatvLoader {
    static LoadAsync(url: string, scene: Scene, camera: Camera | null, onProgress?: (progress: number) => void, useCache?: boolean): Promise<Splatv>;
    static LoadFromFileAsync(file: File, scene: Scene, camera: Camera | null, onProgress?: (progress: number) => void): Promise<Splatv>;
    private static _ParseSplatvBuffer;
}
export { SplatvLoader };
