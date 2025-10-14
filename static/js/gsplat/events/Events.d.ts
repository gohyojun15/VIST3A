import { Object3D } from '../core/Object3D';
declare class ObjectAddedEvent extends Event {
    object: Object3D;
    constructor(object: Object3D);
}
declare class ObjectRemovedEvent extends Event {
    object: Object3D;
    constructor(object: Object3D);
}
declare class ObjectChangedEvent extends Event {
    object: Object3D;
    constructor(object: Object3D);
}
export { ObjectAddedEvent, ObjectRemovedEvent, ObjectChangedEvent };
