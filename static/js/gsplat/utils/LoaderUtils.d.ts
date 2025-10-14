export declare function initiateFetchRequest(url: string, useCache: boolean): Promise<Response>;
export declare function loadDataIntoBuffer(res: Response, onProgress?: (progress: number) => void): Promise<Uint8Array>;
