import { Splat } from '../../../splats/Splat';
import { RenderProgram } from '../programs/RenderProgram';
declare class IntersectionTester {
    testPoint: (x: number, y: number) => Splat | null;
    constructor(renderProgram: RenderProgram, maxDistance?: number, resolution?: number);
}
export { IntersectionTester };
