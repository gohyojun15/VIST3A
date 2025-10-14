import { Scene } from '../core/Scene';
import { Splat } from '../splats/Splat';
declare class PLYLoader {
    static LoadAsync(url: string, scene: Scene, onProgress?: (progress: number) => void, format?: string, useCache?: boolean): Promise<Splat>;
    static LoadFromFileAsync(file: File, scene: Scene, onProgress?: (progress: number) => void, format?: string): Promise<Splat>;
    static LoadFromArrayBuffer(arrayBuffer: ArrayBufferLike, scene: Scene, format?: string): Splat;
    private static _ParsePLYBuffer;
}
export { PLYLoader };
