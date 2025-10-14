class c {
  constructor(t = 0, n = 0, i = 0) {
    this.x = t, this.y = n, this.z = i;
  }
  equals(t) {
    return !(this.x !== t.x || this.y !== t.y || this.z !== t.z);
  }
  add(t) {
    return typeof t == "number" ? new c(this.x + t, this.y + t, this.z + t) : new c(this.x + t.x, this.y + t.y, this.z + t.z);
  }
  subtract(t) {
    return typeof t == "number" ? new c(this.x - t, this.y - t, this.z - t) : new c(this.x - t.x, this.y - t.y, this.z - t.z);
  }
  multiply(t) {
    return typeof t == "number" ? new c(this.x * t, this.y * t, this.z * t) : t instanceof c ? new c(this.x * t.x, this.y * t.y, this.z * t.z) : new c(
      this.x * t.buffer[0] + this.y * t.buffer[4] + this.z * t.buffer[8] + t.buffer[12],
      this.x * t.buffer[1] + this.y * t.buffer[5] + this.z * t.buffer[9] + t.buffer[13],
      this.x * t.buffer[2] + this.y * t.buffer[6] + this.z * t.buffer[10] + t.buffer[14]
    );
  }
  divide(t) {
    return typeof t == "number" ? new c(this.x / t, this.y / t, this.z / t) : new c(this.x / t.x, this.y / t.y, this.z / t.z);
  }
  cross(t) {
    const n = this.y * t.z - this.z * t.y, i = this.z * t.x - this.x * t.z, e = this.x * t.y - this.y * t.x;
    return new c(n, i, e);
  }
  dot(t) {
    return this.x * t.x + this.y * t.y + this.z * t.z;
  }
  lerp(t, n) {
    return new c(this.x + (t.x - this.x) * n, this.y + (t.y - this.y) * n, this.z + (t.z - this.z) * n);
  }
  min(t) {
    return new c(Math.min(this.x, t.x), Math.min(this.y, t.y), Math.min(this.z, t.z));
  }
  max(t) {
    return new c(Math.max(this.x, t.x), Math.max(this.y, t.y), Math.max(this.z, t.z));
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new Error(`Invalid component index: ${t}`);
    }
  }
  minComponent() {
    return this.x < this.y && this.x < this.z ? 0 : this.y < this.z ? 1 : 2;
  }
  maxComponent() {
    return this.x > this.y && this.x > this.z ? 0 : this.y > this.z ? 1 : 2;
  }
  magnitude() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  distanceTo(t) {
    return Math.sqrt((this.x - t.x) ** 2 + (this.y - t.y) ** 2 + (this.z - t.z) ** 2);
  }
  normalize() {
    const t = this.magnitude();
    return new c(this.x / t, this.y / t, this.z / t);
  }
  flat() {
    return [this.x, this.y, this.z];
  }
  clone() {
    return new c(this.x, this.y, this.z);
  }
  toString() {
    return `[${this.flat().join(", ")}]`;
  }
  static One(t = 1) {
    return new c(t, t, t);
  }
}
class b {
  constructor(t = 0, n = 0, i = 0, e = 1) {
    this.x = t, this.y = n, this.z = i, this.w = e;
  }
  equals(t) {
    return !(this.x !== t.x || this.y !== t.y || this.z !== t.z || this.w !== t.w);
  }
  normalize() {
    const t = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
    return new b(this.x / t, this.y / t, this.z / t, this.w / t);
  }
  multiply(t) {
    const n = this.w, i = this.x, e = this.y, Q = this.z, o = t.w, s = t.x, r = t.y, A = t.z;
    return new b(
      n * s + i * o + e * A - Q * r,
      n * r - i * A + e * o + Q * s,
      n * A + i * r - e * s + Q * o,
      n * o - i * s - e * r - Q * A
    );
  }
  inverse() {
    const t = this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
    return new b(-this.x / t, -this.y / t, -this.z / t, this.w / t);
  }
  apply(t) {
    const n = new b(t.x, t.y, t.z, 0), i = new b(-this.x, -this.y, -this.z, this.w), e = this.multiply(n).multiply(i);
    return new c(e.x, e.y, e.z);
  }
  flat() {
    return [this.x, this.y, this.z, this.w];
  }
  clone() {
    return new b(this.x, this.y, this.z, this.w);
  }
  static FromEuler(t) {
    const n = t.x / 2, i = t.y / 2, e = t.z / 2, Q = Math.cos(i), o = Math.sin(i), s = Math.cos(n), r = Math.sin(n), A = Math.cos(e), I = Math.sin(e);
    return new b(
      Q * r * A + o * s * I,
      o * s * A - Q * r * I,
      Q * s * I - o * r * A,
      Q * s * A + o * r * I
    );
  }
  toEuler() {
    const t = 2 * (this.w * this.x + this.y * this.z), n = 1 - 2 * (this.x * this.x + this.y * this.y), i = Math.atan2(t, n);
    let e;
    const Q = 2 * (this.w * this.y - this.z * this.x);
    Math.abs(Q) >= 1 ? e = Math.sign(Q) * Math.PI / 2 : e = Math.asin(Q);
    const o = 2 * (this.w * this.z + this.x * this.y), s = 1 - 2 * (this.y * this.y + this.z * this.z), r = Math.atan2(o, s);
    return new c(i, e, r);
  }
  static FromMatrix3(t) {
    const n = t.buffer, i = n[0] + n[4] + n[8];
    let e, Q, o, s;
    if (i > 0) {
      const r = 0.5 / Math.sqrt(i + 1);
      s = 0.25 / r, e = (n[7] - n[5]) * r, Q = (n[2] - n[6]) * r, o = (n[3] - n[1]) * r;
    } else if (n[0] > n[4] && n[0] > n[8]) {
      const r = 2 * Math.sqrt(1 + n[0] - n[4] - n[8]);
      s = (n[7] - n[5]) / r, e = 0.25 * r, Q = (n[1] + n[3]) / r, o = (n[2] + n[6]) / r;
    } else if (n[4] > n[8]) {
      const r = 2 * Math.sqrt(1 + n[4] - n[0] - n[8]);
      s = (n[2] - n[6]) / r, e = (n[1] + n[3]) / r, Q = 0.25 * r, o = (n[5] + n[7]) / r;
    } else {
      const r = 2 * Math.sqrt(1 + n[8] - n[0] - n[4]);
      s = (n[3] - n[1]) / r, e = (n[2] + n[6]) / r, Q = (n[5] + n[7]) / r, o = 0.25 * r;
    }
    return new b(e, Q, o, s);
  }
  static FromAxisAngle(t, n) {
    const i = n / 2, e = Math.sin(i), Q = Math.cos(i);
    return new b(t.x * e, t.y * e, t.z * e, Q);
  }
  static LookRotation(t) {
    const n = new c(0, 0, 1), i = n.dot(t);
    if (Math.abs(i - -1) < 1e-6)
      return new b(0, 1, 0, Math.PI);
    if (Math.abs(i - 1) < 1e-6)
      return new b();
    const e = Math.acos(i), Q = n.cross(t).normalize();
    return b.FromAxisAngle(Q, e);
  }
  toString() {
    return `[${this.flat().join(", ")}]`;
  }
}
class At {
  constructor() {
    const t = /* @__PURE__ */ new Map();
    this.addEventListener = (n, i) => {
      t.has(n) || t.set(n, /* @__PURE__ */ new Set()), t.get(n).add(i);
    }, this.removeEventListener = (n, i) => {
      t.has(n) && t.get(n).delete(i);
    }, this.hasEventListener = (n, i) => t.has(n) ? t.get(n).has(i) : !1, this.dispatchEvent = (n) => {
      if (t.has(n.type))
        for (const i of t.get(n.type))
          i(n);
    };
  }
}
class z {
  // prettier-ignore
  constructor(t = 1, n = 0, i = 0, e = 0, Q = 0, o = 1, s = 0, r = 0, A = 0, I = 0, d = 1, U = 0, a = 0, B = 0, l = 0, F = 1) {
    this.buffer = [
      t,
      n,
      i,
      e,
      Q,
      o,
      s,
      r,
      A,
      I,
      d,
      U,
      a,
      B,
      l,
      F
    ];
  }
  equals(t) {
    if (this.buffer.length !== t.buffer.length)
      return !1;
    if (this.buffer === t.buffer)
      return !0;
    for (let n = 0; n < this.buffer.length; n++)
      if (this.buffer[n] !== t.buffer[n])
        return !1;
    return !0;
  }
  multiply(t) {
    const n = this.buffer, i = t.buffer;
    return new z(
      i[0] * n[0] + i[1] * n[4] + i[2] * n[8] + i[3] * n[12],
      i[0] * n[1] + i[1] * n[5] + i[2] * n[9] + i[3] * n[13],
      i[0] * n[2] + i[1] * n[6] + i[2] * n[10] + i[3] * n[14],
      i[0] * n[3] + i[1] * n[7] + i[2] * n[11] + i[3] * n[15],
      i[4] * n[0] + i[5] * n[4] + i[6] * n[8] + i[7] * n[12],
      i[4] * n[1] + i[5] * n[5] + i[6] * n[9] + i[7] * n[13],
      i[4] * n[2] + i[5] * n[6] + i[6] * n[10] + i[7] * n[14],
      i[4] * n[3] + i[5] * n[7] + i[6] * n[11] + i[7] * n[15],
      i[8] * n[0] + i[9] * n[4] + i[10] * n[8] + i[11] * n[12],
      i[8] * n[1] + i[9] * n[5] + i[10] * n[9] + i[11] * n[13],
      i[8] * n[2] + i[9] * n[6] + i[10] * n[10] + i[11] * n[14],
      i[8] * n[3] + i[9] * n[7] + i[10] * n[11] + i[11] * n[15],
      i[12] * n[0] + i[13] * n[4] + i[14] * n[8] + i[15] * n[12],
      i[12] * n[1] + i[13] * n[5] + i[14] * n[9] + i[15] * n[13],
      i[12] * n[2] + i[13] * n[6] + i[14] * n[10] + i[15] * n[14],
      i[12] * n[3] + i[13] * n[7] + i[14] * n[11] + i[15] * n[15]
    );
  }
  clone() {
    const t = this.buffer;
    return new z(
      t[0],
      t[1],
      t[2],
      t[3],
      t[4],
      t[5],
      t[6],
      t[7],
      t[8],
      t[9],
      t[10],
      t[11],
      t[12],
      t[13],
      t[14],
      t[15]
    );
  }
  determinant() {
    const t = this.buffer;
    return t[12] * t[9] * t[6] * t[3] - t[8] * t[13] * t[6] * t[3] - t[12] * t[5] * t[10] * t[3] + t[4] * t[13] * t[10] * t[3] + t[8] * t[5] * t[14] * t[3] - t[4] * t[9] * t[14] * t[3] - t[12] * t[9] * t[2] * t[7] + t[8] * t[13] * t[2] * t[7] + t[12] * t[1] * t[10] * t[7] - t[0] * t[13] * t[10] * t[7] - t[8] * t[1] * t[14] * t[7] + t[0] * t[9] * t[14] * t[7] + t[12] * t[5] * t[2] * t[11] - t[4] * t[13] * t[2] * t[11] - t[12] * t[1] * t[6] * t[11] + t[0] * t[13] * t[6] * t[11] + t[4] * t[1] * t[14] * t[11] - t[0] * t[5] * t[14] * t[11] - t[8] * t[5] * t[2] * t[15] + t[4] * t[9] * t[2] * t[15] + t[8] * t[1] * t[6] * t[15] - t[0] * t[9] * t[6] * t[15] - t[4] * t[1] * t[10] * t[15] + t[0] * t[5] * t[10] * t[15];
  }
  invert() {
    const t = this.buffer, n = this.determinant();
    if (n === 0)
      throw new Error("Matrix is not invertible.");
    const i = 1 / n;
    return new z(
      i * (t[5] * t[10] * t[15] - t[5] * t[11] * t[14] - t[9] * t[6] * t[15] + t[9] * t[7] * t[14] + t[13] * t[6] * t[11] - t[13] * t[7] * t[10]),
      i * (-t[1] * t[10] * t[15] + t[1] * t[11] * t[14] + t[9] * t[2] * t[15] - t[9] * t[3] * t[14] - t[13] * t[2] * t[11] + t[13] * t[3] * t[10]),
      i * (t[1] * t[6] * t[15] - t[1] * t[7] * t[14] - t[5] * t[2] * t[15] + t[5] * t[3] * t[14] + t[13] * t[2] * t[7] - t[13] * t[3] * t[6]),
      i * (-t[1] * t[6] * t[11] + t[1] * t[7] * t[10] + t[5] * t[2] * t[11] - t[5] * t[3] * t[10] - t[9] * t[2] * t[7] + t[9] * t[3] * t[6]),
      i * (-t[4] * t[10] * t[15] + t[4] * t[11] * t[14] + t[8] * t[6] * t[15] - t[8] * t[7] * t[14] - t[12] * t[6] * t[11] + t[12] * t[7] * t[10]),
      i * (t[0] * t[10] * t[15] - t[0] * t[11] * t[14] - t[8] * t[2] * t[15] + t[8] * t[3] * t[14] + t[12] * t[2] * t[11] - t[12] * t[3] * t[10]),
      i * (-t[0] * t[6] * t[15] + t[0] * t[7] * t[14] + t[4] * t[2] * t[15] - t[4] * t[3] * t[14] - t[12] * t[2] * t[7] + t[12] * t[3] * t[6]),
      i * (t[0] * t[6] * t[11] - t[0] * t[7] * t[10] - t[4] * t[2] * t[11] + t[4] * t[3] * t[10] + t[8] * t[2] * t[7] - t[8] * t[3] * t[6]),
      i * (t[4] * t[9] * t[15] - t[4] * t[11] * t[13] - t[8] * t[5] * t[15] + t[8] * t[7] * t[13] + t[12] * t[5] * t[11] - t[12] * t[7] * t[9]),
      i * (-t[0] * t[9] * t[15] + t[0] * t[11] * t[13] + t[8] * t[1] * t[15] - t[8] * t[3] * t[13] - t[12] * t[1] * t[11] + t[12] * t[3] * t[9]),
      i * (t[0] * t[5] * t[15] - t[0] * t[7] * t[13] - t[4] * t[1] * t[15] + t[4] * t[3] * t[13] + t[12] * t[1] * t[7] - t[12] * t[3] * t[5]),
      i * (-t[0] * t[5] * t[11] + t[0] * t[7] * t[9] + t[4] * t[1] * t[11] - t[4] * t[3] * t[9] - t[8] * t[1] * t[7] + t[8] * t[3] * t[5]),
      i * (-t[4] * t[9] * t[14] + t[4] * t[10] * t[13] + t[8] * t[5] * t[14] - t[8] * t[6] * t[13] - t[12] * t[5] * t[10] + t[12] * t[6] * t[9]),
      i * (t[0] * t[9] * t[14] - t[0] * t[10] * t[13] - t[8] * t[1] * t[14] + t[8] * t[2] * t[13] + t[12] * t[1] * t[10] - t[12] * t[2] * t[9]),
      i * (-t[0] * t[5] * t[14] + t[0] * t[6] * t[13] + t[4] * t[1] * t[14] - t[4] * t[2] * t[13] - t[12] * t[1] * t[6] + t[12] * t[2] * t[5]),
      i * (t[0] * t[5] * t[10] - t[0] * t[6] * t[9] - t[4] * t[1] * t[10] + t[4] * t[2] * t[9] + t[8] * t[1] * t[6] - t[8] * t[2] * t[5])
    );
  }
  static Compose(t, n, i) {
    const e = n.x, Q = n.y, o = n.z, s = n.w, r = e + e, A = Q + Q, I = o + o, d = e * r, U = e * A, a = e * I, B = Q * A, l = Q * I, F = o * I, C = s * r, h = s * A, V = s * I, u = i.x, S = i.y, y = i.z;
    return new z(
      (1 - (B + F)) * u,
      (U + V) * u,
      (a - h) * u,
      0,
      (U - V) * S,
      (1 - (d + F)) * S,
      (l + C) * S,
      0,
      (a + h) * y,
      (l - C) * y,
      (1 - (d + B)) * y,
      0,
      t.x,
      t.y,
      t.z,
      1
    );
  }
  toString() {
    return `[${this.buffer.join(", ")}]`;
  }
}
class It extends Event {
  constructor(t) {
    super("objectAdded"), this.object = t;
  }
}
class Ct extends Event {
  constructor(t) {
    super("objectRemoved"), this.object = t;
  }
}
class ht extends Event {
  constructor(t) {
    super("objectChanged"), this.object = t;
  }
}
class et extends At {
  constructor() {
    super(), this.positionChanged = !1, this.rotationChanged = !1, this.scaleChanged = !1, this._position = new c(), this._rotation = new b(), this._scale = new c(1, 1, 1), this._transform = new z(), this._changeEvent = new ht(this), this.update = () => {
    }, this.applyPosition = () => {
      this.position = new c();
    }, this.applyRotation = () => {
      this.rotation = new b();
    }, this.applyScale = () => {
      this.scale = new c(1, 1, 1);
    }, this.raiseChangeEvent = () => {
      this.dispatchEvent(this._changeEvent);
    };
  }
  _updateMatrix() {
    this._transform = z.Compose(this._position, this._rotation, this._scale);
  }
  get position() {
    return this._position;
  }
  set position(t) {
    this._position.equals(t) || (this._position = t, this.positionChanged = !0, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
  }
  get rotation() {
    return this._rotation;
  }
  set rotation(t) {
    this._rotation.equals(t) || (this._rotation = t, this.rotationChanged = !0, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
  }
  get scale() {
    return this._scale;
  }
  set scale(t) {
    this._scale.equals(t) || (this._scale = t, this.scaleChanged = !0, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
  }
  get forward() {
    let t = new c(0, 0, 1);
    return t = this.rotation.apply(t), t;
  }
  get transform() {
    return this._transform;
  }
}
class K {
  // prettier-ignore
  constructor(t = 1, n = 0, i = 0, e = 0, Q = 1, o = 0, s = 0, r = 0, A = 1) {
    this.buffer = [
      t,
      n,
      i,
      e,
      Q,
      o,
      s,
      r,
      A
    ];
  }
  equals(t) {
    if (this.buffer.length !== t.buffer.length)
      return !1;
    if (this.buffer === t.buffer)
      return !0;
    for (let n = 0; n < this.buffer.length; n++)
      if (this.buffer[n] !== t.buffer[n])
        return !1;
    return !0;
  }
  multiply(t) {
    const n = this.buffer, i = t.buffer;
    return new K(
      i[0] * n[0] + i[3] * n[1] + i[6] * n[2],
      i[1] * n[0] + i[4] * n[1] + i[7] * n[2],
      i[2] * n[0] + i[5] * n[1] + i[8] * n[2],
      i[0] * n[3] + i[3] * n[4] + i[6] * n[5],
      i[1] * n[3] + i[4] * n[4] + i[7] * n[5],
      i[2] * n[3] + i[5] * n[4] + i[8] * n[5],
      i[0] * n[6] + i[3] * n[7] + i[6] * n[8],
      i[1] * n[6] + i[4] * n[7] + i[7] * n[8],
      i[2] * n[6] + i[5] * n[7] + i[8] * n[8]
    );
  }
  clone() {
    const t = this.buffer;
    return new K(
      t[0],
      t[1],
      t[2],
      t[3],
      t[4],
      t[5],
      t[6],
      t[7],
      t[8]
    );
  }
  static Eye(t = 1) {
    return new K(t, 0, 0, 0, t, 0, 0, 0, t);
  }
  static Diagonal(t) {
    return new K(t.x, 0, 0, 0, t.y, 0, 0, 0, t.z);
  }
  static RotationFromQuaternion(t) {
    return new K(
      1 - 2 * t.y * t.y - 2 * t.z * t.z,
      2 * t.x * t.y - 2 * t.z * t.w,
      2 * t.x * t.z + 2 * t.y * t.w,
      2 * t.x * t.y + 2 * t.z * t.w,
      1 - 2 * t.x * t.x - 2 * t.z * t.z,
      2 * t.y * t.z - 2 * t.x * t.w,
      2 * t.x * t.z - 2 * t.y * t.w,
      2 * t.y * t.z + 2 * t.x * t.w,
      1 - 2 * t.x * t.x - 2 * t.y * t.y
    );
  }
  static RotationFromEuler(t) {
    const n = Math.cos(t.x), i = Math.sin(t.x), e = Math.cos(t.y), Q = Math.sin(t.y), o = Math.cos(t.z), s = Math.sin(t.z), r = [
      e * o + Q * i * s,
      -e * s + Q * i * o,
      Q * n,
      n * s,
      n * o,
      -i,
      -Q * o + e * i * s,
      Q * s + e * i * o,
      e * n
    ];
    return new K(...r);
  }
  toString() {
    return `[${this.buffer.join(", ")}]`;
  }
}
class Y {
  constructor(t = 0, n = null, i = null, e = null, Q = null) {
    this.changed = !1, this.detached = !1, this._vertexCount = t, this._positions = n || new Float32Array(0), this._rotations = i || new Float32Array(0), this._scales = e || new Float32Array(0), this._colors = Q || new Uint8Array(0), this._selection = new Uint8Array(this.vertexCount), this.translate = (o) => {
      for (let s = 0; s < this.vertexCount; s++)
        this.positions[3 * s + 0] += o.x, this.positions[3 * s + 1] += o.y, this.positions[3 * s + 2] += o.z;
      this.changed = !0;
    }, this.rotate = (o) => {
      const s = K.RotationFromQuaternion(o).buffer;
      for (let r = 0; r < this.vertexCount; r++) {
        const A = this.positions[3 * r + 0], I = this.positions[3 * r + 1], d = this.positions[3 * r + 2];
        this.positions[3 * r + 0] = s[0] * A + s[1] * I + s[2] * d, this.positions[3 * r + 1] = s[3] * A + s[4] * I + s[5] * d, this.positions[3 * r + 2] = s[6] * A + s[7] * I + s[8] * d;
        const U = new b(
          this.rotations[4 * r + 1],
          this.rotations[4 * r + 2],
          this.rotations[4 * r + 3],
          this.rotations[4 * r + 0]
        ), a = o.multiply(U);
        this.rotations[4 * r + 1] = a.x, this.rotations[4 * r + 2] = a.y, this.rotations[4 * r + 3] = a.z, this.rotations[4 * r + 0] = a.w;
      }
      this.changed = !0;
    }, this.scale = (o) => {
      for (let s = 0; s < this.vertexCount; s++)
        this.positions[3 * s + 0] *= o.x, this.positions[3 * s + 1] *= o.y, this.positions[3 * s + 2] *= o.z, this.scales[3 * s + 0] *= o.x, this.scales[3 * s + 1] *= o.y, this.scales[3 * s + 2] *= o.z;
      this.changed = !0;
    }, this.serialize = () => {
      const o = new Uint8Array(this.vertexCount * Y.RowLength), s = new Float32Array(o.buffer), r = new Uint8Array(o.buffer);
      for (let A = 0; A < this.vertexCount; A++)
        s[8 * A + 0] = this.positions[3 * A + 0], s[8 * A + 1] = this.positions[3 * A + 1], s[8 * A + 2] = this.positions[3 * A + 2], r[32 * A + 24 + 0] = this.colors[4 * A + 0], r[32 * A + 24 + 1] = this.colors[4 * A + 1], r[32 * A + 24 + 2] = this.colors[4 * A + 2], r[32 * A + 24 + 3] = this.colors[4 * A + 3], s[8 * A + 3 + 0] = this.scales[3 * A + 0], s[8 * A + 3 + 1] = this.scales[3 * A + 1], s[8 * A + 3 + 2] = this.scales[3 * A + 2], r[32 * A + 28 + 0] = this.rotations[4 * A + 0] * 128 + 128 & 255, r[32 * A + 28 + 1] = this.rotations[4 * A + 1] * 128 + 128 & 255, r[32 * A + 28 + 2] = this.rotations[4 * A + 2] * 128 + 128 & 255, r[32 * A + 28 + 3] = this.rotations[4 * A + 3] * 128 + 128 & 255;
      return o;
    }, this.reattach = (o, s, r, A, I) => {
      console.assert(
        o.byteLength === this.vertexCount * 3 * 4,
        `Expected ${this.vertexCount * 3 * 4} bytes, got ${o.byteLength} bytes`
      ), this._positions = new Float32Array(o), this._rotations = new Float32Array(s), this._scales = new Float32Array(r), this._colors = new Uint8Array(A), this._selection = new Uint8Array(I), this.detached = !1;
    };
  }
  static {
    this.RowLength = 3 * 4 + 3 * 4 + 4 + 4;
  }
  static Deserialize(t) {
    const n = t.length / Y.RowLength, i = new Float32Array(3 * n), e = new Float32Array(4 * n), Q = new Float32Array(3 * n), o = new Uint8Array(4 * n), s = new Float32Array(t.buffer), r = new Uint8Array(t.buffer);
    for (let A = 0; A < n; A++)
      i[3 * A + 0] = s[8 * A + 0], i[3 * A + 1] = s[8 * A + 1], i[3 * A + 2] = s[8 * A + 2], e[4 * A + 0] = (r[32 * A + 28 + 0] - 128) / 128, e[4 * A + 1] = (r[32 * A + 28 + 1] - 128) / 128, e[4 * A + 2] = (r[32 * A + 28 + 2] - 128) / 128, e[4 * A + 3] = (r[32 * A + 28 + 3] - 128) / 128, Q[3 * A + 0] = s[8 * A + 3 + 0], Q[3 * A + 1] = s[8 * A + 3 + 1], Q[3 * A + 2] = s[8 * A + 3 + 2], o[4 * A + 0] = r[32 * A + 24 + 0], o[4 * A + 1] = r[32 * A + 24 + 1], o[4 * A + 2] = r[32 * A + 24 + 2], o[4 * A + 3] = r[32 * A + 24 + 3];
    return new Y(n, i, e, Q, o);
  }
  get vertexCount() {
    return this._vertexCount;
  }
  get positions() {
    return this._positions;
  }
  get rotations() {
    return this._rotations;
  }
  get scales() {
    return this._scales;
  }
  get colors() {
    return this._colors;
  }
  get selection() {
    return this._selection;
  }
  clone() {
    return new Y(
      this.vertexCount,
      new Float32Array(this.positions),
      new Float32Array(this.rotations),
      new Float32Array(this.scales),
      new Uint8Array(this.colors)
    );
  }
}
class nt {
  static {
    this.RowLength = 64;
  }
  constructor(t, n, i, e, Q) {
    this._vertexCount = t, this._positions = n, this._data = i, this._width = e, this._height = Q, this.serialize = () => new Uint8Array(this._data.buffer);
  }
  static Deserialize(t, n, i) {
    const e = new Uint32Array(t.buffer), Q = new Float32Array(t.buffer), o = Math.floor(Q.byteLength / this.RowLength), s = new Float32Array(o * 3);
    for (let r = 0; r < o; r++)
      s[3 * r + 0] = Q[16 * r + 0], s[3 * r + 1] = Q[16 * r + 1], s[3 * r + 2] = Q[16 * r + 2], s[3 * r + 0] = Q[16 * r + 3];
    return new nt(o, s, e, n, i);
  }
  get vertexCount() {
    return this._vertexCount;
  }
  get positions() {
    return this._positions;
  }
  get data() {
    return this._data;
  }
  get width() {
    return this._width;
  }
  get height() {
    return this._height;
  }
}
class q {
  static {
    this.SH_C0 = 0.28209479177387814;
  }
  static SplatToPLY(t, n) {
    let i = `ply
format binary_little_endian 1.0
`;
    i += `element vertex ${n}
`;
    const e = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"];
    for (let F = 0; F < 45; F++)
      e.push(`f_rest_${F}`);
    e.push("opacity"), e.push("scale_0"), e.push("scale_1"), e.push("scale_2"), e.push("rot_0"), e.push("rot_1"), e.push("rot_2"), e.push("rot_3");
    for (const F of e)
      i += `property float ${F}
`;
    i += `end_header
`;
    const Q = new TextEncoder().encode(i), o = 4 * 3 + 4 * 3 + 4 * 3 + 4 * 45 + 4 + 4 * 3 + 4 * 4, s = n * o, r = new DataView(new ArrayBuffer(Q.length + s));
    new Uint8Array(r.buffer).set(Q, 0);
    const A = new Float32Array(t), I = new Uint8Array(t), d = Q.length, U = 4 * 3 + 4 * 3, a = U + 4 * 3 + 4 * 45, B = a + 4, l = B + 4 * 3;
    for (let F = 0; F < n; F++) {
      const C = A[8 * F + 0], h = A[8 * F + 1], V = A[8 * F + 2], u = (I[32 * F + 24 + 0] / 255 - 0.5) / this.SH_C0, S = (I[32 * F + 24 + 1] / 255 - 0.5) / this.SH_C0, y = (I[32 * F + 24 + 2] / 255 - 0.5) / this.SH_C0, Z = I[32 * F + 24 + 3] / 255, M = Math.log(Z / (1 - Z)), J = Math.log(A[8 * F + 3 + 0]), P = Math.log(A[8 * F + 3 + 1]), v = Math.log(A[8 * F + 3 + 2]);
      let L = new b(
        (I[32 * F + 28 + 1] - 128) / 128,
        (I[32 * F + 28 + 2] - 128) / 128,
        (I[32 * F + 28 + 3] - 128) / 128,
        (I[32 * F + 28 + 0] - 128) / 128
      );
      L = L.normalize();
      const m = L.w, x = L.x, G = L.y, D = L.z;
      r.setFloat32(d + o * F + 0, C, !0), r.setFloat32(d + o * F + 4, h, !0), r.setFloat32(d + o * F + 8, V, !0), r.setFloat32(d + o * F + U + 0, u, !0), r.setFloat32(d + o * F + U + 4, S, !0), r.setFloat32(d + o * F + U + 8, y, !0), r.setFloat32(d + o * F + a, M, !0), r.setFloat32(d + o * F + B + 0, J, !0), r.setFloat32(d + o * F + B + 4, P, !0), r.setFloat32(d + o * F + B + 8, v, !0), r.setFloat32(d + o * F + l + 0, m, !0), r.setFloat32(d + o * F + l + 4, x, !0), r.setFloat32(d + o * F + l + 8, G, !0), r.setFloat32(d + o * F + l + 12, D, !0);
    }
    return r.buffer;
  }
}
class $ {
  constructor(t, n) {
    this.min = t, this.max = n;
  }
  contains(t) {
    return t.x >= this.min.x && t.x <= this.max.x && t.y >= this.min.y && t.y <= this.max.y && t.z >= this.min.z && t.z <= this.max.z;
  }
  intersects(t) {
    return this.max.x >= t.min.x && this.min.x <= t.max.x && this.max.y >= t.min.y && this.min.y <= t.max.y && this.max.z >= t.min.z && this.min.z <= t.max.z;
  }
  size() {
    return this.max.subtract(this.min);
  }
  center() {
    return this.min.add(this.max).divide(2);
  }
  expand(t) {
    this.min = this.min.min(t), this.max = this.max.max(t);
  }
  permute() {
    const t = this.min, n = this.max;
    this.min = new c(Math.min(t.x, n.x), Math.min(t.y, n.y), Math.min(t.z, n.z)), this.max = new c(Math.max(t.x, n.x), Math.max(t.y, n.y), Math.max(t.z, n.z));
  }
}
class j extends et {
  constructor(t = void 0) {
    super(), this.selectedChanged = !1, this.colorTransformChanged = !1, this._selected = !1, this._colorTransforms = [], this._colorTransformsMap = /* @__PURE__ */ new Map(), this._data = t || new Y(), this._bounds = new $(
      new c(1 / 0, 1 / 0, 1 / 0),
      new c(-1 / 0, -1 / 0, -1 / 0)
    ), this.recalculateBounds = () => {
      this._bounds = new $(
        new c(1 / 0, 1 / 0, 1 / 0),
        new c(-1 / 0, -1 / 0, -1 / 0)
      );
      for (let n = 0; n < this._data.vertexCount; n++)
        this._bounds.expand(
          new c(
            this._data.positions[3 * n],
            this._data.positions[3 * n + 1],
            this._data.positions[3 * n + 2]
          )
        );
    }, this.applyPosition = () => {
      this.data.translate(this.position), this.position = new c();
    }, this.applyRotation = () => {
      this.data.rotate(this.rotation), this.rotation = new b();
    }, this.applyScale = () => {
      this.data.scale(this.scale), this.scale = new c(1, 1, 1);
    }, this.recalculateBounds();
  }
  saveToFile(t = null, n = "splat") {
    if (!document) return;
    if (!t) {
      const s = /* @__PURE__ */ new Date();
      t = `splat-${s.getFullYear()}-${s.getMonth() + 1}-${s.getDate()}.${n}`;
    }
    const i = this.clone();
    i.applyRotation(), i.applyScale(), i.applyPosition();
    const e = i.data.serialize();
    let Q;
    if (n === "ply") {
      const s = q.SplatToPLY(e.buffer, i.data.vertexCount);
      Q = new Blob([s], { type: "application/octet-stream" });
    } else
      Q = new Blob([e.buffer], { type: "application/octet-stream" });
    const o = document.createElement("a");
    o.download = t, o.href = URL.createObjectURL(Q), o.click();
  }
  get data() {
    return this._data;
  }
  get selected() {
    return this._selected;
  }
  set selected(t) {
    this._selected !== t && (this._selected = t, this.selectedChanged = !0, this.dispatchEvent(this._changeEvent));
  }
  get colorTransforms() {
    return this._colorTransforms;
  }
  get colorTransformsMap() {
    return this._colorTransformsMap;
  }
  get bounds() {
    let t = this._bounds.center();
    t = t.add(this.position);
    let n = this._bounds.size();
    return n = n.multiply(this.scale), new $(t.subtract(n.divide(2)), t.add(n.divide(2)));
  }
  clone() {
    const t = new j(this.data.clone());
    return t.position = this.position.clone(), t.rotation = this.rotation.clone(), t.scale = this.scale.clone(), t;
  }
}
class O extends et {
  constructor(t) {
    super(), this._data = t;
  }
  get data() {
    return this._data;
  }
}
class ct {
  constructor() {
    this._fx = 1132, this._fy = 1132, this._near = 0.1, this._far = 100, this._width = 512, this._height = 512, this._projectionMatrix = new z(), this._viewMatrix = new z(), this._viewProj = new z(), this._updateProjectionMatrix = () => {
      this._projectionMatrix = new z(
        2 * this.fx / this.width,
        0,
        0,
        0,
        0,
        -2 * this.fy / this.height,
        0,
        0,
        0,
        0,
        this.far / (this.far - this.near),
        1,
        0,
        0,
        -(this.far * this.near) / (this.far - this.near),
        0
      ), this._viewProj = this.projectionMatrix.multiply(this.viewMatrix);
    }, this.update = (t, n) => {
      const i = K.RotationFromQuaternion(n).buffer, e = t.flat();
      this._viewMatrix = new z(
        i[0],
        i[1],
        i[2],
        0,
        i[3],
        i[4],
        i[5],
        0,
        i[6],
        i[7],
        i[8],
        0,
        -e[0] * i[0] - e[1] * i[3] - e[2] * i[6],
        -e[0] * i[1] - e[1] * i[4] - e[2] * i[7],
        -e[0] * i[2] - e[1] * i[5] - e[2] * i[8],
        1
      ), this._viewProj = this.projectionMatrix.multiply(this.viewMatrix);
    }, this.setSize = (t, n) => {
      this._width = t, this._height = n, this._updateProjectionMatrix();
    };
  }
  get fx() {
    return this._fx;
  }
  set fx(t) {
    this._fx !== t && (this._fx = t, this._updateProjectionMatrix());
  }
  get fy() {
    return this._fy;
  }
  set fy(t) {
    this._fy !== t && (this._fy = t, this._updateProjectionMatrix());
  }
  get near() {
    return this._near;
  }
  set near(t) {
    this._near !== t && (this._near = t, this._updateProjectionMatrix());
  }
  get far() {
    return this._far;
  }
  set far(t) {
    this._far !== t && (this._far = t, this._updateProjectionMatrix());
  }
  get width() {
    return this._width;
  }
  get height() {
    return this._height;
  }
  get projectionMatrix() {
    return this._projectionMatrix;
  }
  get viewMatrix() {
    return this._viewMatrix;
  }
  get viewProj() {
    return this._viewProj;
  }
}
class H {
  constructor(t = 0, n = 0, i = 0, e = 0) {
    this.x = t, this.y = n, this.z = i, this.w = e;
  }
  equals(t) {
    return !(this.x !== t.x || this.y !== t.y || this.z !== t.z || this.w !== t.w);
  }
  add(t) {
    return typeof t == "number" ? new H(this.x + t, this.y + t, this.z + t, this.w + t) : new H(this.x + t.x, this.y + t.y, this.z + t.z, this.w + t.w);
  }
  subtract(t) {
    return typeof t == "number" ? new H(this.x - t, this.y - t, this.z - t, this.w - t) : new H(this.x - t.x, this.y - t.y, this.z - t.z, this.w - t.w);
  }
  multiply(t) {
    return typeof t == "number" ? new H(this.x * t, this.y * t, this.z * t, this.w * t) : t instanceof H ? new H(this.x * t.x, this.y * t.y, this.z * t.z, this.w * t.w) : new H(
      this.x * t.buffer[0] + this.y * t.buffer[4] + this.z * t.buffer[8] + this.w * t.buffer[12],
      this.x * t.buffer[1] + this.y * t.buffer[5] + this.z * t.buffer[9] + this.w * t.buffer[13],
      this.x * t.buffer[2] + this.y * t.buffer[6] + this.z * t.buffer[10] + this.w * t.buffer[14],
      this.x * t.buffer[3] + this.y * t.buffer[7] + this.z * t.buffer[11] + this.w * t.buffer[15]
    );
  }
  dot(t) {
    return this.x * t.x + this.y * t.y + this.z * t.z + this.w * t.w;
  }
  lerp(t, n) {
    return new H(
      this.x + (t.x - this.x) * n,
      this.y + (t.y - this.y) * n,
      this.z + (t.z - this.z) * n,
      this.w + (t.w - this.w) * n
    );
  }
  magnitude() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
  }
  distanceTo(t) {
    return Math.sqrt((this.x - t.x) ** 2 + (this.y - t.y) ** 2 + (this.z - t.z) ** 2 + (this.w - t.w) ** 2);
  }
  normalize() {
    const t = this.magnitude();
    return new H(this.x / t, this.y / t, this.z / t, this.w / t);
  }
  flat() {
    return [this.x, this.y, this.z, this.w];
  }
  clone() {
    return new H(this.x, this.y, this.z, this.w);
  }
  toString() {
    return `[${this.flat().join(", ")}]`;
  }
}
class yt extends et {
  constructor(t = void 0) {
    super(), this._data = t || new ct(), this._position = new c(0, 0, -5), this.update = () => {
      this.data.update(this.position, this.rotation);
    }, this.screenPointToRay = (n, i) => {
      const e = new H(n, i, -1, 1), Q = this._data.projectionMatrix.invert(), o = e.multiply(Q), s = this._data.viewMatrix.invert(), r = o.multiply(s);
      return new c(
        r.x / r.w,
        r.y / r.w,
        r.z / r.w
      ).subtract(this.position).normalize();
    };
  }
  get data() {
    return this._data;
  }
}
class wt extends At {
  constructor() {
    super(), this._objects = [], this.addObject = (t) => {
      this.objects.push(t), this.dispatchEvent(new It(t));
    }, this.removeObject = (t) => {
      const n = this.objects.indexOf(t);
      if (n < 0)
        throw new Error("Object not found in scene");
      this.objects.splice(n, 1), this.dispatchEvent(new Ct(t));
    }, this.findObject = (t) => {
      for (const n of this.objects)
        if (t(n))
          return n;
    }, this.findObjectOfType = (t) => {
      for (const n of this.objects)
        if (n instanceof t)
          return n;
    }, this.reset = () => {
      const t = this.objects.slice();
      for (const n of t)
        this.removeObject(n);
    }, this.reset();
  }
  getMergedSceneDataBuffer(t = "splat") {
    const n = [];
    let i = 0;
    for (const o of this.objects)
      if (o instanceof j) {
        const s = o.clone();
        s.applyRotation(), s.applyScale(), s.applyPosition();
        const r = s.data.serialize();
        n.push(r), i += s.data.vertexCount;
      }
    const e = new Uint8Array(i * Y.RowLength);
    let Q = 0;
    for (const o of n)
      e.set(o, Q), Q += o.length;
    return t === "ply" ? q.SplatToPLY(e.buffer, i) : e.buffer;
  }
  saveToFile(t = null, n = "splat") {
    if (!document) return;
    if (!t) {
      const o = /* @__PURE__ */ new Date();
      t = `scene-${o.getFullYear()}-${o.getMonth() + 1}-${o.getDate()}.${n}`;
    }
    const i = this.getMergedSceneDataBuffer(n), e = new Blob([i], { type: "application/octet-stream" }), Q = document.createElement("a");
    Q.download = t, Q.href = URL.createObjectURL(e), Q.click();
  }
  get objects() {
    return this._objects;
  }
}
async function it(E, t) {
  const n = await fetch(E, {
    mode: "cors",
    credentials: "omit",
    cache: t ? "force-cache" : "default"
  });
  if (n.status != 200)
    throw new Error(n.status + " Unable to load " + n.url);
  return n;
}
async function st(E, t) {
  const n = E.body.getReader(), i = E.headers.get("content-length"), e = i && !isNaN(parseInt(i)) ? parseInt(i) : void 0, Q = [];
  let o = 0;
  for (; ; ) {
    const { done: A, value: I } = await n.read();
    if (A) break;
    if (Q.push(I), o += I.length, t && e) {
      const d = o / e, U = Math.min(d * 0.95, 0.95);
      t(U);
    }
  }
  const s = new Uint8Array(o);
  let r = 0;
  for (const A of Q)
    s.set(A, r), r += A.length;
  return t && t(1), s;
}
class kt {
  static async LoadAsync(t, n, i, e = !1) {
    const Q = await it(t, e), o = await st(Q, i);
    return this.LoadFromArrayBuffer(o.buffer, n);
  }
  static async LoadFromFileAsync(t, n, i) {
    const e = new FileReader();
    let Q = new j();
    return e.onload = (o) => {
      Q = this.LoadFromArrayBuffer(o.target.result, n);
    }, e.onprogress = (o) => {
      i?.(o.loaded / o.total);
    }, e.readAsArrayBuffer(t), await new Promise((o) => {
      e.onloadend = () => {
        o();
      };
    }), Q;
  }
  static LoadFromArrayBuffer(t, n) {
    const i = new Uint8Array(t), e = Y.Deserialize(i), Q = new j(e);
    return n.addObject(Q), Q;
  }
}
class Tt {
  static async LoadAsync(t, n, i, e = "", Q = !1) {
    const o = await it(t, Q), s = await st(o, i);
    if (s[0] !== 112 || s[1] !== 108 || s[2] !== 121 || s[3] !== 10)
      throw new Error("Invalid PLY file");
    return this.LoadFromArrayBuffer(s.buffer, n, e);
  }
  static async LoadFromFileAsync(t, n, i, e = "") {
    const Q = new FileReader();
    let o = new j();
    return Q.onload = (s) => {
      o = this.LoadFromArrayBuffer(s.target.result, n, e);
    }, Q.onprogress = (s) => {
      i?.(s.loaded / s.total);
    }, Q.readAsArrayBuffer(t), await new Promise((s) => {
      Q.onloadend = () => {
        s();
      };
    }), o;
  }
  static LoadFromArrayBuffer(t, n, i = "") {
    const e = new Uint8Array(this._ParsePLYBuffer(t, i)), Q = Y.Deserialize(e), o = new j(Q);
    return n.addObject(o), o;
  }
  static _ParsePLYBuffer(t, n) {
    const i = new Uint8Array(t), e = new TextDecoder().decode(i.slice(0, 1024 * 10)), Q = `end_header
`, o = e.indexOf(Q);
    if (o < 0) throw new Error("Unable to read .ply file header");
    const s = parseInt(/element vertex (\d+)\n/.exec(e)[1]);
    let r = 0;
    const A = {
      double: 8,
      int: 4,
      uint: 4,
      float: 4,
      short: 2,
      ushort: 2,
      uchar: 1
    }, I = [];
    for (const B of e.slice(0, o).split(`
`).filter((l) => l.startsWith("property "))) {
      const [l, F, C] = B.split(" ");
      if (I.push({ name: C, type: F, offset: r }), !A[F]) throw new Error(`Unsupported property type: ${F}`);
      r += A[F];
    }
    const d = new DataView(t, o + Q.length), U = new ArrayBuffer(Y.RowLength * s), a = b.FromEuler(new c(Math.PI / 2, 0, 0));
    for (let B = 0; B < s; B++) {
      const l = new Float32Array(U, B * Y.RowLength, 3), F = new Float32Array(U, B * Y.RowLength + 12, 3), C = new Uint8ClampedArray(U, B * Y.RowLength + 24, 4), h = new Uint8ClampedArray(U, B * Y.RowLength + 28, 4);
      let V = 255, u = 0, S = 0, y = 0;
      I.forEach((M) => {
        let J;
        switch (M.type) {
          case "float":
            J = d.getFloat32(M.offset + B * r, !0);
            break;
          case "int":
            J = d.getInt32(M.offset + B * r, !0);
            break;
          default:
            throw new Error(`Unsupported property type: ${M.type}`);
        }
        switch (M.name) {
          case "x":
            l[0] = J;
            break;
          case "y":
            l[1] = J;
            break;
          case "z":
            l[2] = J;
            break;
          case "scale_0":
          case "scaling_0":
            F[0] = Math.exp(J);
            break;
          case "scale_1":
          case "scaling_1":
            F[1] = Math.exp(J);
            break;
          case "scale_2":
          case "scaling_2":
            F[2] = Math.exp(J);
            break;
          case "red":
            C[0] = J;
            break;
          case "green":
            C[1] = J;
            break;
          case "blue":
            C[2] = J;
            break;
          case "f_dc_0":
          case "features_0":
            C[0] = (0.5 + q.SH_C0 * J) * 255;
            break;
          case "f_dc_1":
          case "features_1":
            C[1] = (0.5 + q.SH_C0 * J) * 255;
            break;
          case "f_dc_2":
          case "features_2":
            C[2] = (0.5 + q.SH_C0 * J) * 255;
            break;
          case "f_dc_3":
            C[3] = (0.5 + q.SH_C0 * J) * 255;
            break;
          case "opacity":
          case "opacity_0":
            C[3] = 1 / (1 + Math.exp(-J)) * 255;
            break;
          case "rot_0":
          case "rotation_0":
            V = J;
            break;
          case "rot_1":
          case "rotation_1":
            u = J;
            break;
          case "rot_2":
          case "rotation_2":
            S = J;
            break;
          case "rot_3":
          case "rotation_3":
            y = J;
            break;
        }
      });
      let Z = new b(u, S, y, V);
      switch (n) {
        case "polycam": {
          const M = l[1];
          l[1] = -l[2], l[2] = M, Z = a.multiply(Z);
          break;
        }
        case "":
          break;
        default:
          throw new Error(`Unsupported format: ${n}`);
      }
      Z = Z.normalize(), h[0] = Z.w * 128 + 128, h[1] = Z.x * 128 + 128, h[2] = Z.y * 128 + 128, h[3] = Z.z * 128 + 128;
    }
    return U;
  }
}
class bt {
  static async LoadAsync(t, n, i, e, Q = !1) {
    const o = await it(t, Q), s = await st(o, e);
    return this._ParseSplatvBuffer(s.buffer, n, i);
  }
  static async LoadFromFileAsync(t, n, i, e) {
    const Q = new FileReader();
    let o = null;
    if (Q.onload = (s) => {
      o = this._ParseSplatvBuffer(s.target.result, n, i);
    }, Q.onprogress = (s) => {
      e?.(s.loaded / s.total);
    }, Q.readAsArrayBuffer(t), await new Promise((s) => {
      Q.onloadend = () => {
        s();
      };
    }), !o)
      throw new Error("Failed to load splatv file");
    return o;
  }
  static _ParseSplatvBuffer(t, n, i) {
    let e = null;
    const Q = (U, a, B) => {
      if (U.type === "magic") {
        const l = new Int32Array(a.buffer);
        if (l[0] !== 26443)
          throw new Error("Invalid splatv file");
        B.push({ size: l[1], type: "chunks" });
      } else if (U.type === "chunks") {
        const l = JSON.parse(new TextDecoder("utf-8").decode(a));
        if (l.length == 0)
          throw new Error("Invalid splatv file");
        l.length > 1 && console.warn("Splatv file contains more than one chunk, only the first one will be loaded");
        const F = l[0], C = F.cameras;
        if (i && C && C.length) {
          const h = C[0], V = new c(
            h.position[0],
            h.position[1],
            h.position[2]
          ), u = b.FromMatrix3(
            new K(
              h.rotation[0][0],
              h.rotation[0][1],
              h.rotation[0][2],
              h.rotation[1][0],
              h.rotation[1][1],
              h.rotation[1][2],
              h.rotation[2][0],
              h.rotation[2][1],
              h.rotation[2][2]
            )
          );
          i.position = V, i.rotation = u;
        }
        B.push(F);
      } else if (U.type === "splat") {
        const l = nt.Deserialize(a, U.texwidth, U.texheight), F = new O(l);
        n.addObject(F), e = F;
      }
    }, o = new Uint8Array(t), s = [
      { size: 8, type: "magic", texwidth: 0, texheight: 0 }
    ];
    let r = s.shift(), A = new Uint8Array(r.size), I = 0, d = 0;
    for (; r; ) {
      for (; I < r.size; ) {
        const U = Math.min(r.size - I, o.length - d);
        A.set(o.subarray(d, d + U), I), I += U, d += U;
      }
      if (Q(r, A, s), e)
        return e;
      r = s.shift(), r && (A = new Uint8Array(r.size), I = 0);
    }
    throw new Error("Invalid splatv file");
  }
}
const Ut = "YXN5bmMgZnVuY3Rpb24gckEoQyA9IHt9KSB7CiAgdmFyIHIsIEkgPSBDLCBZID0gaW1wb3J0Lm1ldGEudXJsLCBUID0gIiIsIHU7CiAgewogICAgdHJ5IHsKICAgICAgVCA9IG5ldyBVUkwoIi4iLCBZKS5ocmVmOwogICAgfSBjYXRjaCB7CiAgICB9CiAgICB1ID0gKEEpID0+IHsKICAgICAgdmFyIEIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTsKICAgICAgcmV0dXJuIEIub3BlbigiR0VUIiwgQSwgITEpLCBCLnJlc3BvbnNlVHlwZSA9ICJhcnJheWJ1ZmZlciIsIEIuc2VuZChudWxsKSwgbmV3IFVpbnQ4QXJyYXkoQi5yZXNwb25zZSk7CiAgICB9OwogIH0KICBjb25zb2xlLmxvZy5iaW5kKGNvbnNvbGUpLCBjb25zb2xlLmVycm9yLmJpbmQoY29uc29sZSk7CiAgdmFyIGQsIEosIGwsIHAsIEggPSAhMTsKICBmdW5jdGlvbiBrKCkgewogICAgdmFyIEEgPSBsLmJ1ZmZlcjsKICAgIEkuSEVBUFU4ID0gcCA9IG5ldyBVaW50OEFycmF5KEEpLCBJLkhFQVBVMzIgPSBuZXcgVWludDMyQXJyYXkoQSksIEkuSEVBUEYzMiA9IG5ldyBGbG9hdDMyQXJyYXkoQSksIG5ldyBCaWdJbnQ2NEFycmF5KEEpLCBuZXcgQmlnVWludDY0QXJyYXkoQSk7CiAgfQogIGZ1bmN0aW9uIFgoKSB7CiAgICBpZiAoSS5wcmVSdW4pCiAgICAgIGZvciAodHlwZW9mIEkucHJlUnVuID09ICJmdW5jdGlvbiIgJiYgKEkucHJlUnVuID0gW0kucHJlUnVuXSk7IEkucHJlUnVuLmxlbmd0aDsgKQogICAgICAgIEFBKEkucHJlUnVuLnNoaWZ0KCkpOwogICAgcSh2KTsKICB9CiAgZnVuY3Rpb24gUCgpIHsKICAgIEggPSAhMCwgRC5jKCk7CiAgfQogIGZ1bmN0aW9uIGIoKSB7CiAgICBpZiAoSS5wb3N0UnVuKQogICAgICBmb3IgKHR5cGVvZiBJLnBvc3RSdW4gPT0gImZ1bmN0aW9uIiAmJiAoSS5wb3N0UnVuID0gW0kucG9zdFJ1bl0pOyBJLnBvc3RSdW4ubGVuZ3RoOyApCiAgICAgICAgJChJLnBvc3RSdW4uc2hpZnQoKSk7CiAgICBxKHgpOwogIH0KICB2YXIgUzsKICBmdW5jdGlvbiBfKCkgewogICAgcmV0dXJuIElBKCJBR0Z6YlFFQUFBQUJHUVJnQVg4QmYyQUJmd0JnQ1g5L2YzOS9mMzkvZndCZ0FBQUNCd0VCWVFGaEFBQURCZ1VBQVFBQ0F3VUhBUUdDQW9DQUFnWUlBWDhCUVlDTUJBc0hGUVVCWWdJQUFXTUFCUUZrQUFRQlpRQURBV1lBQWd3QkFRcTRPQVZVQWdGL0FYNENRRUdBQ0NnQ0FDSUJyU0FBclVJSGZFTDQvLy8vSDROOElnSkMvLy8vL3c5WUJFQWdBcWNpQUQ4QVFSQjBUUTBCSUFBUUFBMEJDMEdFQ0VFd05nSUFRWDhQQzBHQUNDQUFOZ0lBSUFFTDN3c0JDSDhDUUNBQVJRMEFJQUJCQ0dzaUF5QUFRUVJyS0FJQUlnSkJlSEVpQUdvaEJRSkFJQUpCQVhFTkFDQUNRUUp4UlEwQklBTWdBeWdDQUNJRWF5SURRWmdJS0FJQVNRMEJJQUFnQkdvaEFBSkFBa0FDUUVHY0NDZ0NBQ0FEUndSQUlBTW9BZ3doQVNBRVFmOEJUUVJBSUFFZ0F5Z0NDQ0lDUncwQ1FZZ0lRWWdJS0FJQVFYNGdCRUVEZG5keE5nSUFEQVVMSUFNb0FoZ2hCeUFCSUFOSEJFQWdBeWdDQ0NJQ0lBRTJBZ3dnQVNBQ05nSUlEQVFMSUFNb0FoUWlBZ1IvSUFOQkZHb0ZJQU1vQWhBaUFrVU5BeUFEUVJCcUN5RUVBMEFnQkNFR0lBSWlBVUVVYWlFRUlBRW9BaFFpQWcwQUlBRkJFR29oQkNBQktBSVFJZ0lOQUFzZ0JrRUFOZ0lBREFNTElBVW9BZ1FpQWtFRGNVRURSdzBEUVpBSUlBQTJBZ0FnQlNBQ1FYNXhOZ0lFSUFNZ0FFRUJjallDQkNBRklBQTJBZ0FQQ3lBQ0lBRTJBZ3dnQVNBQ05nSUlEQUlMUVFBaEFRc2dCMFVOQUFKQUlBTW9BaHdpQkVFQ2RDSUNLQUs0Q2lBRFJnUkFJQUpCdUFwcUlBRTJBZ0FnQVEwQlFZd0lRWXdJS0FJQVFYNGdCSGR4TmdJQURBSUxBa0FnQXlBSEtBSVFSZ1JBSUFjZ0FUWUNFQXdCQ3lBSElBRTJBaFFMSUFGRkRRRUxJQUVnQnpZQ0dDQURLQUlRSWdJRVFDQUJJQUkyQWhBZ0FpQUJOZ0lZQ3lBREtBSVVJZ0pGRFFBZ0FTQUNOZ0lVSUFJZ0FUWUNHQXNnQXlBRlR3MEFJQVVvQWdRaUJFRUJjVVVOQUFKQUFrQUNRQUpBSUFSQkFuRkZCRUJCb0Fnb0FnQWdCVVlFUUVHZ0NDQUROZ0lBUVpRSVFaUUlLQUlBSUFCcUlnQTJBZ0FnQXlBQVFRRnlOZ0lFSUFOQm5BZ29BZ0JIRFFaQmtBaEJBRFlDQUVHY0NFRUFOZ0lBRHd0Qm5BZ29BZ0FpQnlBRlJnUkFRWndJSUFNMkFnQkJrQWhCa0Fnb0FnQWdBR29pQURZQ0FDQURJQUJCQVhJMkFnUWdBQ0FEYWlBQU5nSUFEd3NnQkVGNGNTQUFhaUVBSUFVb0Fnd2hBU0FFUWY4QlRRUkFJQVVvQWdnaUFpQUJSZ1JBUVlnSVFZZ0lLQUlBUVg0Z0JFRURkbmR4TmdJQURBVUxJQUlnQVRZQ0RDQUJJQUkyQWdnTUJBc2dCU2dDR0NFSUlBRWdCVWNFUUNBRktBSUlJZ0lnQVRZQ0RDQUJJQUkyQWdnTUF3c2dCU2dDRkNJQ0JIOGdCVUVVYWdVZ0JTZ0NFQ0lDUlEwQ0lBVkJFR29MSVFRRFFDQUVJUVlnQWlJQlFSUnFJUVFnQVNnQ0ZDSUNEUUFnQVVFUWFpRUVJQUVvQWhBaUFnMEFDeUFHUVFBMkFnQU1BZ3NnQlNBRVFYNXhOZ0lFSUFNZ0FFRUJjallDQkNBQUlBTnFJQUEyQWdBTUF3dEJBQ0VCQ3lBSVJRMEFBa0FnQlNnQ0hDSUVRUUowSWdJb0FyZ0tJQVZHQkVBZ0FrRzRDbW9nQVRZQ0FDQUJEUUZCakFoQmpBZ29BZ0JCZmlBRWQzRTJBZ0FNQWdzQ1FDQUZJQWdvQWhCR0JFQWdDQ0FCTmdJUURBRUxJQWdnQVRZQ0ZBc2dBVVVOQVFzZ0FTQUlOZ0lZSUFVb0FoQWlBZ1JBSUFFZ0FqWUNFQ0FDSUFFMkFoZ0xJQVVvQWhRaUFrVU5BQ0FCSUFJMkFoUWdBaUFCTmdJWUN5QURJQUJCQVhJMkFnUWdBQ0FEYWlBQU5nSUFJQU1nQjBjTkFFR1FDQ0FBTmdJQUR3c2dBRUgvQVUwRVFDQUFRZmdCY1VHd0NHb2hBZ0ovUVlnSUtBSUFJZ1JCQVNBQVFRTjJkQ0lBY1VVRVFFR0lDQ0FBSUFSeU5nSUFJQUlNQVFzZ0FpZ0NDQXNoQUNBQ0lBTTJBZ2dnQUNBRE5nSU1JQU1nQWpZQ0RDQURJQUEyQWdnUEMwRWZJUUVnQUVILy8vOEhUUVJBSUFCQkppQUFRUWgyWnlJQ2EzWkJBWEVnQWtFQmRHdEJQbW9oQVFzZ0F5QUJOZ0ljSUFOQ0FEY0NFQ0FCUVFKMFFiZ0thaUVFQW44Q1FBSi9RWXdJS0FJQUlnWkJBU0FCZENJQ2NVVUVRRUdNQ0NBQ0lBWnlOZ0lBSUFRZ0F6WUNBRUVZSVFGQkNBd0JDeUFBUVJrZ0FVRUJkbXRCQUNBQlFSOUhHM1FoQVNBRUtBSUFJUVFEUUNBRUlnSW9BZ1JCZUhFZ0FFWU5BaUFCUVIxMklRUWdBVUVCZENFQklBSWdCRUVFY1dvaUJpZ0NFQ0lFRFFBTElBWWdBellDRUVFWUlRRWdBaUVFUVFnTElRQWdBeUlDREFFTElBSW9BZ2dpQkNBRE5nSU1JQUlnQXpZQ0NFRVlJUUJCQ0NFQlFRQUxJUVlnQVNBRGFpQUVOZ0lBSUFNZ0FqWUNEQ0FBSUFOcUlBWTJBZ0JCcUFoQnFBZ29BZ0JCQVdzaUFFRi9JQUFiTmdJQUN3dTRKd0VMZnlNQVFSQnJJZ29rQUFKQUFrQUNRQUpBQWtBQ1FBSkFBa0FDUUFKQUlBQkI5QUZOQkVCQmlBZ29BZ0FpQkVFUUlBQkJDMnBCK0FOeElBQkJDMGtiSWdaQkEzWWlBSFlpQVVFRGNRUkFBa0FnQVVGL2MwRUJjU0FBYWlJRFFRTjBJZ0ZCc0FocUlnQWdBU2dDdUFnaUFpZ0NDQ0lGUmdSQVFZZ0lJQVJCZmlBRGQzRTJBZ0FNQVFzZ0JTQUFOZ0lNSUFBZ0JUWUNDQXNnQWtFSWFpRUFJQUlnQVVFRGNqWUNCQ0FCSUFKcUlnRWdBU2dDQkVFQmNqWUNCQXdMQ3lBR1FaQUlLQUlBSWdoTkRRRWdBUVJBQWtCQkFpQUFkQ0lDUVFBZ0FtdHlJQUVnQUhSeGFDSURRUU4wSWdGQnNBaHFJZ0lnQVNnQ3VBZ2lBQ2dDQ0NJRlJnUkFRWWdJSUFSQmZpQURkM0VpQkRZQ0FBd0JDeUFGSUFJMkFnd2dBaUFGTmdJSUN5QUFJQVpCQTNJMkFnUWdBQ0FHYWlJSElBRWdCbXNpQlVFQmNqWUNCQ0FBSUFGcUlBVTJBZ0FnQ0FSQUlBaEJlSEZCc0FocUlRRkJuQWdvQWdBaEFnSi9JQVJCQVNBSVFRTjJkQ0lEY1VVRVFFR0lDQ0FESUFSeU5nSUFJQUVNQVFzZ0FTZ0NDQXNoQXlBQklBSTJBZ2dnQXlBQ05nSU1JQUlnQVRZQ0RDQUNJQU0yQWdnTElBQkJDR29oQUVHY0NDQUhOZ0lBUVpBSUlBVTJBZ0FNQ3d0QmpBZ29BZ0FpQzBVTkFTQUxhRUVDZENnQ3VBb2lBaWdDQkVGNGNTQUdheUVESUFJaEFRTkFBa0FnQVNnQ0VDSUFSUVJBSUFFb0FoUWlBRVVOQVFzZ0FDZ0NCRUY0Y1NBR2F5SUJJQU1nQVNBRFNTSUJHeUVESUFBZ0FpQUJHeUVDSUFBaEFRd0JDd3NnQWlnQ0dDRUpJQUlnQWlnQ0RDSUFSd1JBSUFJb0FnZ2lBU0FBTmdJTUlBQWdBVFlDQ0F3S0N5QUNLQUlVSWdFRWZ5QUNRUlJxQlNBQ0tBSVFJZ0ZGRFFNZ0FrRVFhZ3NoQlFOQUlBVWhCeUFCSWdCQkZHb2hCU0FBS0FJVUlnRU5BQ0FBUVJCcUlRVWdBQ2dDRUNJQkRRQUxJQWRCQURZQ0FBd0pDMEYvSVFZZ0FFRy9mMHNOQUNBQVFRdHFJZ0ZCZUhFaEJrR01DQ2dDQUNJSFJRMEFRUjhoQ0VFQUlBWnJJUU1nQUVIMC8vOEhUUVJBSUFaQkppQUJRUWgyWnlJQWEzWkJBWEVnQUVFQmRHdEJQbW9oQ0FzQ1FBSkFBa0FnQ0VFQ2RDZ0N1QW9pQVVVRVFFRUFJUUFNQVF0QkFDRUFJQVpCR1NBSVFRRjJhMEVBSUFoQkgwY2JkQ0VDQTBBQ1FDQUJLQUlFUVhoeElBWnJJZ1FnQTA4TkFDQUJJUVVnQkNJRERRQkJBQ0VESUFFaEFBd0RDeUFBSUFFb0FoUWlCQ0FFSUFFZ0FrRWRka0VFY1dvb0FoQWlBVVliSUFBZ0JCc2hBQ0FDUVFGMElRSWdBUTBBQ3dzZ0FDQUZja1VFUUVFQUlRVkJBaUFJZENJQVFRQWdBR3R5SUFkeElnQkZEUU1nQUdoQkFuUW9BcmdLSVFBTElBQkZEUUVMQTBBZ0FDZ0NCRUY0Y1NBR2F5SUNJQU5KSVFFZ0FpQURJQUViSVFNZ0FDQUZJQUViSVFVZ0FDZ0NFQ0lCQkg4Z0FRVWdBQ2dDRkFzaUFBMEFDd3NnQlVVTkFDQURRWkFJS0FJQUlBWnJUdzBBSUFVb0FoZ2hDQ0FGSUFVb0Fnd2lBRWNFUUNBRktBSUlJZ0VnQURZQ0RDQUFJQUUyQWdnTUNBc2dCU2dDRkNJQkJIOGdCVUVVYWdVZ0JTZ0NFQ0lCUlEwRElBVkJFR29MSVFJRFFDQUNJUVFnQVNJQVFSUnFJUUlnQUNnQ0ZDSUJEUUFnQUVFUWFpRUNJQUFvQWhBaUFRMEFDeUFFUVFBMkFnQU1Cd3NnQmtHUUNDZ0NBQ0lGVFFSQVFad0lLQUlBSVFBQ1FDQUZJQVpySWdGQkVFOEVRQ0FBSUFacUlnSWdBVUVCY2pZQ0JDQUFJQVZxSUFFMkFnQWdBQ0FHUVFOeU5nSUVEQUVMSUFBZ0JVRURjallDQkNBQUlBVnFJZ0VnQVNnQ0JFRUJjallDQkVFQUlRSkJBQ0VCQzBHUUNDQUJOZ0lBUVp3SUlBSTJBZ0FnQUVFSWFpRUFEQWtMSUFaQmxBZ29BZ0FpQWtrRVFFR1VDQ0FDSUFacklnRTJBZ0JCb0FoQm9BZ29BZ0FpQUNBR2FpSUNOZ0lBSUFJZ0FVRUJjallDQkNBQUlBWkJBM0kyQWdRZ0FFRUlhaUVBREFrTFFRQWhBQ0FHUVM5cUlnTUNmMEhnQ3lnQ0FBUkFRZWdMS0FJQURBRUxRZXdMUW44M0FnQkI1QXRDZ0tDQWdJQ0FCRGNDQUVIZ0N5QUtRUXhxUVhCeFFkaXExYW9GY3pZQ0FFSDBDMEVBTmdJQVFjUUxRUUEyQWdCQmdDQUxJZ0ZxSWdSQkFDQUJheUlIY1NJQklBWk5EUWhCd0Fzb0FnQWlCUVJBUWJnTEtBSUFJZ2dnQVdvaUNTQUlUU0FGSUFsSmNnMEpDd0pBUWNRTExRQUFRUVJ4UlFSQUFrQUNRQUpBQWtCQm9BZ29BZ0FpQlFSQVFjZ0xJUUFEUUNBQUtBSUFJZ2dnQlUwRVFDQUZJQWdnQUNnQ0JHcEpEUU1MSUFBb0FnZ2lBQTBBQ3d0QkFCQUJJZ0pCZjBZTkF5QUJJUVJCNUFzb0FnQWlBRUVCYXlJRklBSnhCRUFnQVNBQ2F5QUNJQVZxUVFBZ0FHdHhhaUVFQ3lBRUlBWk5EUU5Cd0Fzb0FnQWlBQVJBUWJnTEtBSUFJZ1VnQkdvaUJ5QUZUU0FBSUFkSmNnMEVDeUFFRUFFaUFDQUNSdzBCREFVTElBUWdBbXNnQjNFaUJCQUJJZ0lnQUNnQ0FDQUFLQUlFYWtZTkFTQUNJUUFMSUFCQmYwWU5BU0FHUVRCcUlBUk5CRUFnQUNFQ0RBUUxRZWdMS0FJQUlnSWdBeUFFYTJwQkFDQUNhM0VpQWhBQlFYOUdEUUVnQWlBRWFpRUVJQUFoQWd3REN5QUNRWDlIRFFJTFFjUUxRY1FMS0FJQVFRUnlOZ0lBQ3lBQkVBRWlBa0YvUmtFQUVBRWlBRUYvUm5JZ0FDQUNUWElOQlNBQUlBSnJJZ1FnQmtFb2FrME5CUXRCdUF0QnVBc29BZ0FnQkdvaUFEWUNBRUc4Q3lnQ0FDQUFTUVJBUWJ3TElBQTJBZ0FMQWtCQm9BZ29BZ0FpQXdSQVFjZ0xJUUFEUUNBQ0lBQW9BZ0FpQVNBQUtBSUVJZ1ZxUmcwQ0lBQW9BZ2dpQUEwQUN3d0VDMEdZQ0NnQ0FDSUFRUUFnQUNBQ1RSdEZCRUJCbUFnZ0FqWUNBQXRCQUNFQVFjd0xJQVEyQWdCQnlBc2dBallDQUVHb0NFRi9OZ0lBUWF3SVFlQUxLQUlBTmdJQVFkUUxRUUEyQWdBRFFDQUFRUU4wSWdFZ0FVR3dDR29pQlRZQ3VBZ2dBU0FGTmdLOENDQUFRUUZxSWdCQklFY05BQXRCbEFnZ0JFRW9heUlBUVhnZ0FtdEJCM0VpQVdzaUJUWUNBRUdnQ0NBQklBSnFJZ0UyQWdBZ0FTQUZRUUZ5TmdJRUlBQWdBbXBCS0RZQ0JFR2tDRUh3Q3lnQ0FEWUNBQXdFQ3lBQ0lBTk5JQUVnQTB0eURRSWdBQ2dDREVFSWNRMENJQUFnQkNBRmFqWUNCRUdnQ0NBRFFYZ2dBMnRCQjNFaUFHb2lBVFlDQUVHVUNFR1VDQ2dDQUNBRWFpSUNJQUJySWdBMkFnQWdBU0FBUVFGeU5nSUVJQUlnQTJwQktEWUNCRUdrQ0VId0N5Z0NBRFlDQUF3REMwRUFJUUFNQmd0QkFDRUFEQVFMUVpnSUtBSUFJQUpMQkVCQm1BZ2dBallDQUFzZ0FpQUVhaUVGUWNnTElRQUNRQU5BSUFVZ0FDZ0NBQ0lCUndSQUlBQW9BZ2dpQUEwQkRBSUxDeUFBTFFBTVFRaHhSUTBEQzBISUN5RUFBMEFDUUNBQUtBSUFJZ0VnQTAwRVFDQURJQUVnQUNnQ0JHb2lCVWtOQVFzZ0FDZ0NDQ0VBREFFTEMwR1VDQ0FFUVNocklnQkJlQ0FDYTBFSGNTSUJheUlITmdJQVFhQUlJQUVnQW1vaUFUWUNBQ0FCSUFkQkFYSTJBZ1FnQUNBQ2FrRW9OZ0lFUWFRSVFmQUxLQUlBTmdJQUlBTWdCVUVuSUFWclFRZHhha0V2YXlJQUlBQWdBMEVRYWtrYklnRkJHellDQkNBQlFkQUxLUUlBTndJUUlBRkJ5QXNwQWdBM0FnaEIwQXNnQVVFSWFqWUNBRUhNQ3lBRU5nSUFRY2dMSUFJMkFnQkIxQXRCQURZQ0FDQUJRUmhxSVFBRFFDQUFRUWMyQWdRZ0FFRUlhaUFBUVFScUlRQWdCVWtOQUFzZ0FTQURSZzBBSUFFZ0FTZ0NCRUYrY1RZQ0JDQURJQUVnQTJzaUFrRUJjallDQkNBQklBSTJBZ0FDZnlBQ1FmOEJUUVJBSUFKQitBRnhRYkFJYWlFQUFuOUJpQWdvQWdBaUFVRUJJQUpCQTNaMElnSnhSUVJBUVlnSUlBRWdBbkkyQWdBZ0FBd0JDeUFBS0FJSUN5RUJJQUFnQXpZQ0NDQUJJQU0yQWd4QkRDRUNRUWdNQVF0Qkh5RUFJQUpCLy8vL0IwMEVRQ0FDUVNZZ0FrRUlkbWNpQUd0MlFRRnhJQUJCQVhSclFUNXFJUUFMSUFNZ0FEWUNIQ0FEUWdBM0FoQWdBRUVDZEVHNENtb2hBUUpBQWtCQmpBZ29BZ0FpQlVFQklBQjBJZ1J4UlFSQVFZd0lJQVFnQlhJMkFnQWdBU0FETmdJQURBRUxJQUpCR1NBQVFRRjJhMEVBSUFCQkgwY2JkQ0VBSUFFb0FnQWhCUU5BSUFVaUFTZ0NCRUY0Y1NBQ1JnMENJQUJCSFhZaEJTQUFRUUYwSVFBZ0FTQUZRUVJ4YWlJRUtBSVFJZ1VOQUFzZ0JDQUROZ0lRQ3lBRElBRTJBaGhCQ0NFQ0lBTWlBU0VBUVF3TUFRc2dBU2dDQ0NJQUlBTTJBZ3dnQVNBRE5nSUlJQU1nQURZQ0NFRUFJUUJCR0NFQ1FRd0xJQU5xSUFFMkFnQWdBaUFEYWlBQU5nSUFDMEdVQ0NnQ0FDSUFJQVpORFFCQmxBZ2dBQ0FHYXlJQk5nSUFRYUFJUWFBSUtBSUFJZ0FnQm1vaUFqWUNBQ0FDSUFGQkFYSTJBZ1FnQUNBR1FRTnlOZ0lFSUFCQkNHb2hBQXdFQzBHRUNFRXdOZ0lBUVFBaEFBd0RDeUFBSUFJMkFnQWdBQ0FBS0FJRUlBUnFOZ0lFSUFKQmVDQUNhMEVIY1dvaUNDQUdRUU55TmdJRUlBRkJlQ0FCYTBFSGNXb2lCQ0FHSUFocUlnTnJJUWNDUUVHZ0NDZ0NBQ0FFUmdSQVFhQUlJQU0yQWdCQmxBaEJsQWdvQWdBZ0Iyb2lBRFlDQUNBRElBQkJBWEkyQWdRTUFRdEJuQWdvQWdBZ0JFWUVRRUdjQ0NBRE5nSUFRWkFJUVpBSUtBSUFJQWRxSWdBMkFnQWdBeUFBUVFGeU5nSUVJQUFnQTJvZ0FEWUNBQXdCQ3lBRUtBSUVJZ0JCQTNGQkFVWUVRQ0FBUVhoeElRa2dCQ2dDRENFQ0FrQWdBRUgvQVUwRVFDQUVLQUlJSWdFZ0FrWUVRRUdJQ0VHSUNDZ0NBRUYrSUFCQkEzWjNjVFlDQUF3Q0N5QUJJQUkyQWd3Z0FpQUJOZ0lJREFFTElBUW9BaGdoQmdKQUlBSWdCRWNFUUNBRUtBSUlJZ0FnQWpZQ0RDQUNJQUEyQWdnTUFRc0NRQ0FFS0FJVUlnQUVmeUFFUVJScUJTQUVLQUlRSWdCRkRRRWdCRUVRYWdzaEFRTkFJQUVoQlNBQUlnSkJGR29oQVNBQUtBSVVJZ0FOQUNBQ1FSQnFJUUVnQWlnQ0VDSUFEUUFMSUFWQkFEWUNBQXdCQzBFQUlRSUxJQVpGRFFBQ1FDQUVLQUljSWdCQkFuUWlBU2dDdUFvZ0JFWUVRQ0FCUWJnS2FpQUNOZ0lBSUFJTkFVR01DRUdNQ0NnQ0FFRitJQUIzY1RZQ0FBd0NDd0pBSUFRZ0JpZ0NFRVlFUUNBR0lBSTJBaEFNQVFzZ0JpQUNOZ0lVQ3lBQ1JRMEJDeUFDSUFZMkFoZ2dCQ2dDRUNJQUJFQWdBaUFBTmdJUUlBQWdBallDR0FzZ0JDZ0NGQ0lBUlEwQUlBSWdBRFlDRkNBQUlBSTJBaGdMSUFjZ0NXb2hCeUFFSUFscUlnUW9BZ1FoQUFzZ0JDQUFRWDV4TmdJRUlBTWdCMEVCY2pZQ0JDQURJQWRxSUFjMkFnQWdCMEgvQVUwRVFDQUhRZmdCY1VHd0NHb2hBQUovUVlnSUtBSUFJZ0ZCQVNBSFFRTjJkQ0lDY1VVRVFFR0lDQ0FCSUFKeU5nSUFJQUFNQVFzZ0FDZ0NDQXNoQVNBQUlBTTJBZ2dnQVNBRE5nSU1JQU1nQURZQ0RDQURJQUUyQWdnTUFRdEJIeUVDSUFkQi8vLy9CMDBFUUNBSFFTWWdCMEVJZG1jaUFHdDJRUUZ4SUFCQkFYUnJRVDVxSVFJTElBTWdBallDSENBRFFnQTNBaEFnQWtFQ2RFRzRDbW9oQUFKQUFrQkJqQWdvQWdBaUFVRUJJQUowSWdWeFJRUkFRWXdJSUFFZ0JYSTJBZ0FnQUNBRE5nSUFEQUVMSUFkQkdTQUNRUUYyYTBFQUlBSkJIMGNiZENFQ0lBQW9BZ0FoQVFOQUlBRWlBQ2dDQkVGNGNTQUhSZzBDSUFKQkhYWWhBU0FDUVFGMElRSWdBQ0FCUVFSeGFpSUZLQUlRSWdFTkFBc2dCU0FETmdJUUN5QURJQUEyQWhnZ0F5QUROZ0lNSUFNZ0F6WUNDQXdCQ3lBQUtBSUlJZ0VnQXpZQ0RDQUFJQU0yQWdnZ0EwRUFOZ0lZSUFNZ0FEWUNEQ0FESUFFMkFnZ0xJQWhCQ0dvaEFBd0NDd0pBSUFoRkRRQUNRQ0FGS0FJY0lnRkJBblFpQWlnQ3VBb2dCVVlFUUNBQ1FiZ0thaUFBTmdJQUlBQU5BVUdNQ0NBSFFYNGdBWGR4SWdjMkFnQU1BZ3NDUUNBRklBZ29BaEJHQkVBZ0NDQUFOZ0lRREFFTElBZ2dBRFlDRkFzZ0FFVU5BUXNnQUNBSU5nSVlJQVVvQWhBaUFRUkFJQUFnQVRZQ0VDQUJJQUEyQWhnTElBVW9BaFFpQVVVTkFDQUFJQUUyQWhRZ0FTQUFOZ0lZQ3dKQUlBTkJEMDBFUUNBRklBTWdCbW9pQUVFRGNqWUNCQ0FBSUFWcUlnQWdBQ2dDQkVFQmNqWUNCQXdCQ3lBRklBWkJBM0kyQWdRZ0JTQUdhaUlFSUFOQkFYSTJBZ1FnQXlBRWFpQUROZ0lBSUFOQi93Rk5CRUFnQTBINEFYRkJzQWhxSVFBQ2YwR0lDQ2dDQUNJQlFRRWdBMEVEZG5RaUFuRkZCRUJCaUFnZ0FTQUNjallDQUNBQURBRUxJQUFvQWdnTElRRWdBQ0FFTmdJSUlBRWdCRFlDRENBRUlBQTJBZ3dnQkNBQk5nSUlEQUVMUVI4aEFDQURRZi8vL3dkTkJFQWdBMEVtSUFOQkNIWm5JZ0JyZGtFQmNTQUFRUUYwYTBFK2FpRUFDeUFFSUFBMkFod2dCRUlBTndJUUlBQkJBblJCdUFwcUlRRUNRQUpBSUFkQkFTQUFkQ0lDY1VVRVFFR01DQ0FDSUFkeU5nSUFJQUVnQkRZQ0FDQUVJQUUyQWhnTUFRc2dBMEVaSUFCQkFYWnJRUUFnQUVFZlJ4dDBJUUFnQVNnQ0FDRUJBMEFnQVNJQ0tBSUVRWGh4SUFOR0RRSWdBRUVkZGlFQklBQkJBWFFoQUNBQ0lBRkJCSEZxSWdjb0FoQWlBUTBBQ3lBSElBUTJBaEFnQkNBQ05nSVlDeUFFSUFRMkFnd2dCQ0FFTmdJSURBRUxJQUlvQWdnaUFDQUVOZ0lNSUFJZ0JEWUNDQ0FFUVFBMkFoZ2dCQ0FDTmdJTUlBUWdBRFlDQ0FzZ0JVRUlhaUVBREFFTEFrQWdDVVVOQUFKQUlBSW9BaHdpQVVFQ2RDSUZLQUs0Q2lBQ1JnUkFJQVZCdUFwcUlBQTJBZ0FnQUEwQlFZd0lJQXRCZmlBQmQzRTJBZ0FNQWdzQ1FDQUNJQWtvQWhCR0JFQWdDU0FBTmdJUURBRUxJQWtnQURZQ0ZBc2dBRVVOQVFzZ0FDQUpOZ0lZSUFJb0FoQWlBUVJBSUFBZ0FUWUNFQ0FCSUFBMkFoZ0xJQUlvQWhRaUFVVU5BQ0FBSUFFMkFoUWdBU0FBTmdJWUN3SkFJQU5CRDAwRVFDQUNJQU1nQm1vaUFFRURjallDQkNBQUlBSnFJZ0FnQUNnQ0JFRUJjallDQkF3QkN5QUNJQVpCQTNJMkFnUWdBaUFHYWlJRklBTkJBWEkyQWdRZ0F5QUZhaUFETmdJQUlBZ0VRQ0FJUVhoeFFiQUlhaUVBUVp3SUtBSUFJUUVDZjBFQklBaEJBM1owSWdjZ0JIRkZCRUJCaUFnZ0JDQUhjallDQUNBQURBRUxJQUFvQWdnTElRUWdBQ0FCTmdJSUlBUWdBVFlDRENBQklBQTJBZ3dnQVNBRU5nSUlDMEdjQ0NBRk5nSUFRWkFJSUFNMkFnQUxJQUpCQ0dvaEFBc2dDa0VRYWlRQUlBQUx3Z1FDQm44S2ZVSC8vLy8vQnlFTVFZQ0FnSUI0SVExQmZ5RUpBMEFnQXlBS1JnUkFRUUFoQUNBSVFRQkJnSUFRL0FzQVF3RC9mMGNnRFNBTWE3S1ZJUThGSUFRZ0NrRU1iR29pQ3lvQ0FDRVRJQXNxQWdnaEZDQUxLZ0lFSVJVZ0NTQUNJQXBCQW5RaURtb29BZ0FpQzBjRVFDQUJJQXRCMEFCc2FpSUpLZ0k4SUFBcUFqZ2lENVFnQ1NvQ09DQUFLZ0lvSWhDVUlBa3FBakFnQUNvQ0NDSVJsQ0FBS2dJWUloSWdDU29DTkpTU2twSWhGaUFKS2dJc0lBK1VJQWtxQWlnZ0VKUWdDU29DSUNBUmxDQVNJQWtxQWlTVWtwS1NJUmNnQ1NvQ0hDQVBsQ0FKS2dJWUlCQ1VJQWtxQWhBZ0VaUWdFaUFKS2dJVWxKS1NraUVZSUFrcUFnd2dENVFnQ1NvQ0NDQVFsQ0FKS2dJQUlCR1VJQWtxQWdRZ0VwU1NrcEloRHlBTElRa0xJQVVnRG1vZ0ZpQVhJQlNVSUE4Z0U1UWdGU0FZbEpLU2trTUFBSUJGbFB3QUlnczJBZ0FnRENBTElBc2dERW9iSVF3Z0RTQUxJQXNnRFVnYklRMGdDa0VCYWlFS0RBRUxDd05BSUFBZ0EwWkZCRUFnQlNBQVFRSjBhaUlCSUE4Z0FTZ0NBQ0FNYTdPVS9BRWlBVFlDQUNBSUlBRkJBblJxSWdFZ0FTZ0NBRUVCYWpZQ0FDQUFRUUZxSVFBTUFRc0xRUUFoQUNBSFFRQTJBZ0JCQUNFTVFRRWhDZ05BSUFwQmdJQUVSZ1JBQTBBQ1FDQUFJQU5HRFFBZ0J5QUZJQUJCQW5ScUtBSUFRUUowYWlJQklBRW9BZ0FpQVVFQmFqWUNBQ0FHSUFGQkFuUnFJQUEyQWdBZ0FFRUJhaUVBREFFTEN3VWdCeUFLUVFKMElnRnFJQUVnQ0dwQkJHc29BZ0FnREdvaUREWUNBQ0FLUVFGcUlRb01BUXNMQ3dJQUN3c0pBUUJCZ1FnTEFnWUIiKTsKICB9CiAgZnVuY3Rpb24gVyhBKSB7CiAgICBpZiAoQXJyYXlCdWZmZXIuaXNWaWV3KEEpKQogICAgICByZXR1cm4gQTsKICAgIGlmIChBID09IFMgJiYgZCkKICAgICAgcmV0dXJuIG5ldyBVaW50OEFycmF5KGQpOwogICAgaWYgKHUpCiAgICAgIHJldHVybiB1KEEpOwogICAgdGhyb3cgJ3N5bmMgZmV0Y2hpbmcgb2YgdGhlIHdhc20gZmFpbGVkOiB5b3UgY2FuIHByZWxvYWQgaXQgdG8gTW9kdWxlWyJ3YXNtQmluYXJ5Il0gbWFudWFsbHksIG9yIGVtY2MucHkgd2lsbCBkbyB0aGF0IGZvciB5b3Ugd2hlbiBnZW5lcmF0aW5nIEhUTUwgKGJ1dCBub3QgSlMpJzsKICB9CiAgZnVuY3Rpb24gVihBLCBCKSB7CiAgICB2YXIgdCwgRSA9IFcoQSk7CiAgICB0ID0gbmV3IFdlYkFzc2VtYmx5Lk1vZHVsZShFKTsKICAgIHZhciBpID0gbmV3IFdlYkFzc2VtYmx5Lkluc3RhbmNlKHQsIEIpOwogICAgcmV0dXJuIFtpLCB0XTsKICB9CiAgZnVuY3Rpb24geigpIHsKICAgIHJldHVybiB7IGE6IHRBIH07CiAgfQogIGZ1bmN0aW9uIE8oKSB7CiAgICBmdW5jdGlvbiBBKEUsIGkpIHsKICAgICAgcmV0dXJuIEQgPSBFLmV4cG9ydHMsIGwgPSBELmIsIGsoKSwgRUEoRCksIEQ7CiAgICB9CiAgICB2YXIgQiA9IHooKTsKICAgIGlmIChJLmluc3RhbnRpYXRlV2FzbSkKICAgICAgcmV0dXJuIG5ldyBQcm9taXNlKChFLCBpKSA9PiB7CiAgICAgICAgSS5pbnN0YW50aWF0ZVdhc20oQiwgKG4sIHMpID0+IHsKICAgICAgICAgIEUoQShuKSk7CiAgICAgICAgfSk7CiAgICAgIH0pOwogICAgUyA/Pz0gXygpOwogICAgdmFyIHQgPSBWKFMsIEIpOwogICAgcmV0dXJuIEEodFswXSk7CiAgfQogIGZvciAodmFyIHEgPSAoQSkgPT4gewogICAgZm9yICg7IEEubGVuZ3RoID4gMDsgKQogICAgICBBLnNoaWZ0KCkoSSk7CiAgfSwgeCA9IFtdLCAkID0gKEEpID0+IHgucHVzaChBKSwgdiA9IFtdLCBBQSA9IChBKSA9PiB2LnB1c2goQSksIElBID0gKEEpID0+IHsKICAgIGZvciAodmFyIEIsIHQsIEUgPSAwLCBpID0gMCwgbiA9IEEubGVuZ3RoLCBzID0gbmV3IFVpbnQ4QXJyYXkoKG4gKiAzID4+IDIpIC0gKEFbbiAtIDJdID09ICI9IikgLSAoQVtuIC0gMV0gPT0gIj0iKSk7IEUgPCBuOyBFICs9IDQsIGkgKz0gMykKICAgICAgQiA9IGFbQS5jaGFyQ29kZUF0KEUgKyAxKV0sIHQgPSBhW0EuY2hhckNvZGVBdChFICsgMildLCBzW2ldID0gYVtBLmNoYXJDb2RlQXQoRSldIDw8IDIgfCBCID4+IDQsIHNbaSArIDFdID0gQiA8PCA0IHwgdCA+PiAyLCBzW2kgKyAyXSA9IHQgPDwgNiB8IGFbQS5jaGFyQ29kZUF0KEUgKyAzKV07CiAgICByZXR1cm4gczsKICB9LCBnQSA9ICgpID0+IDIxNDc0ODM2NDgsIENBID0gKEEsIEIpID0+IE1hdGguY2VpbChBIC8gQikgKiBCLCBCQSA9IChBKSA9PiB7CiAgICB2YXIgQiA9IGwuYnVmZmVyLmJ5dGVMZW5ndGgsIHQgPSAoQSAtIEIgKyA2NTUzNSkgLyA2NTUzNiB8IDA7CiAgICB0cnkgewogICAgICByZXR1cm4gbC5ncm93KHQpLCBrKCksIDE7CiAgICB9IGNhdGNoIHsKICAgIH0KICB9LCBRQSA9IChBKSA9PiB7CiAgICB2YXIgQiA9IHAubGVuZ3RoOwogICAgQSA+Pj49IDA7CiAgICB2YXIgdCA9IGdBKCk7CiAgICBpZiAoQSA+IHQpCiAgICAgIHJldHVybiAhMTsKICAgIGZvciAodmFyIEUgPSAxOyBFIDw9IDQ7IEUgKj0gMikgewogICAgICB2YXIgaSA9IEIgKiAoMSArIDAuMiAvIEUpOwogICAgICBpID0gTWF0aC5taW4oaSwgQSArIDEwMDY2MzI5Nik7CiAgICAgIHZhciBuID0gTWF0aC5taW4odCwgQ0EoTWF0aC5tYXgoQSwgaSksIDY1NTM2KSksIHMgPSBCQShuKTsKICAgICAgaWYgKHMpCiAgICAgICAgcmV0dXJuICEwOwogICAgfQogICAgcmV0dXJuICExOwogIH0sIGEgPSBuZXcgVWludDhBcnJheSgxMjMpLCBlID0gMjU7IGUgPj0gMDsgLS1lKQogICAgYVs0OCArIGVdID0gNTIgKyBlLCBhWzY1ICsgZV0gPSBlLCBhWzk3ICsgZV0gPSAyNiArIGU7CiAgaWYgKGFbNDNdID0gNjIsIGFbNDddID0gNjMsIEkubm9FeGl0UnVudGltZSAmJiBJLm5vRXhpdFJ1bnRpbWUsIEkucHJpbnQgJiYgSS5wcmludCwgSS5wcmludEVyciAmJiBJLnByaW50RXJyLCBJLndhc21CaW5hcnkgJiYgKGQgPSBJLndhc21CaW5hcnkpLCBJLmFyZ3VtZW50cyAmJiBJLmFyZ3VtZW50cywgSS50aGlzUHJvZ3JhbSAmJiBJLnRoaXNQcm9ncmFtLCBJLnByZUluaXQpCiAgICBmb3IgKHR5cGVvZiBJLnByZUluaXQgPT0gImZ1bmN0aW9uIiAmJiAoSS5wcmVJbml0ID0gW0kucHJlSW5pdF0pOyBJLnByZUluaXQubGVuZ3RoID4gMDsgKQogICAgICBJLnByZUluaXQuc2hpZnQoKSgpOwogIGZ1bmN0aW9uIEVBKEEpIHsKICAgIEkuX3NvcnQgPSBBLmQsIEkuX21hbGxvYyA9IEEuZSwgSS5fZnJlZSA9IEEuZjsKICB9CiAgdmFyIHRBID0geyBhOiBRQSB9OwogIGZ1bmN0aW9uIGlBKCkgewogICAgWCgpOwogICAgZnVuY3Rpb24gQSgpIHsKICAgICAgSS5jYWxsZWRSdW4gPSAhMCwgUCgpLCBKPy4oSSksIEkub25SdW50aW1lSW5pdGlhbGl6ZWQ/LigpLCBiKCk7CiAgICB9CiAgICBJLnNldFN0YXR1cyA/IChJLnNldFN0YXR1cygiUnVubmluZy4uLiIpLCBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgc2V0VGltZW91dCgoKSA9PiBJLnNldFN0YXR1cygiIiksIDEpLCBBKCk7CiAgICB9LCAxKSkgOiBBKCk7CiAgfQogIHZhciBEOwogIHJldHVybiBEID0gTygpLCBpQSgpLCBIID8gciA9IEkgOiByID0gbmV3IFByb21pc2UoKEEsIEIpID0+IHsKICAgIEogPSBBOwogIH0pLCByOwp9CmxldCBnLCBRLCBVLCBmLCBOLCB3LCBNLCB5LCBtLCBMLCBvID0gMCwgUiA9IDAsIEsgPSBbXSwgRiA9ICEwLCBoID0gITEsIGMgPSAhMSwgRyA9ICExOwphc3luYyBmdW5jdGlvbiBhQSgpIHsKICBpZiAoIWcgJiYgKGcgPSBhd2FpdCByQSgpLCAhZyB8fCAhZy5IRUFQRjMyIHx8ICFnLl9zb3J0KSkKICAgIHRocm93IG5ldyBFcnJvcigiV0FTTSBtb2R1bGUgZmFpbGVkIHRvIGluaXRpYWxpemUgcHJvcGVybHkiKTsKfQpjb25zdCBaID0gYXN5bmMgKCkgPT4gewogIGlmIChoKSB7CiAgICBjID0gITA7CiAgICByZXR1cm47CiAgfQogIGggPSAhMCwgYyA9ICExLCBnIHx8IGF3YWl0IGFBKCk7CiAgY29uc3QgQyA9IE1hdGgucG93KDIsIE1hdGguY2VpbChNYXRoLmxvZzIoUS52ZXJ0ZXhDb3VudCkpKTsKICBvIDwgQyAmJiAobyA+IDAgJiYgKGcuX2ZyZWUoVSksIGcuX2ZyZWUoTiksIGcuX2ZyZWUodyksIGcuX2ZyZWUoTSksIGcuX2ZyZWUoeSksIGcuX2ZyZWUobSksIGcuX2ZyZWUoTCkpLCBvID0gQywgVSA9IGcuX21hbGxvYygxNiAqIDQpLCBOID0gZy5fbWFsbG9jKG8gKiA0KSwgdyA9IGcuX21hbGxvYygzICogbyAqIDQpLCBNID0gZy5fbWFsbG9jKG8gKiA0KSwgeSA9IGcuX21hbGxvYyhvICogNCksIG0gPSBnLl9tYWxsb2MobyAqIDQpLCBMID0gZy5fbWFsbG9jKG8gKiA0KSksIFIgPCBRLnRyYW5zZm9ybXMubGVuZ3RoICYmIChSID4gMCAmJiBnLl9mcmVlKGYpLCBSID0gUS50cmFuc2Zvcm1zLmxlbmd0aCwgZiA9IGcuX21hbGxvYyhSICogNCkpLCBoID0gITEsIGMgJiYgKGMgPSAhMSwgYXdhaXQgWigpKTsKfSwgb0EgPSAoKSA9PiB7CiAgaWYgKCEoaCB8fCBjIHx8ICFnIHx8ICFRKSkgewogICAgaCA9ICEwOwogICAgdHJ5IHsKICAgICAgY29uc3QgQyA9IGcuSEVBUEYzMiwgciA9IGcuSEVBUFUzMjsKICAgICAgaWYgKHcgLyA0ICsgUS5wb3NpdGlvbnMubGVuZ3RoID4gQy5sZW5ndGgpCiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCJQb3NpdGlvbnMgYnVmZmVyIG92ZXJmbG93Iik7CiAgICAgIGlmIChmIC8gNCArIFEudHJhbnNmb3Jtcy5sZW5ndGggPiBDLmxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIlRyYW5zZm9ybXMgYnVmZmVyIG92ZXJmbG93Iik7CiAgICAgIGlmIChOIC8gNCArIFEudHJhbnNmb3JtSW5kaWNlcy5sZW5ndGggPiByLmxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIlRyYW5zZm9ybSBpbmRpY2VzIGJ1ZmZlciBvdmVyZmxvdyIpOwogICAgICBpZiAoQy5zZXQoUS5wb3NpdGlvbnMsIHcgLyA0KSwgQy5zZXQoUS50cmFuc2Zvcm1zLCBmIC8gNCksIHIuc2V0KFEudHJhbnNmb3JtSW5kaWNlcywgTiAvIDQpLCBDLnNldChuZXcgRmxvYXQzMkFycmF5KEspLCBVIC8gNCksIGcuX3NvcnQoCiAgICAgICAgVSwKICAgICAgICBmLAogICAgICAgIE4sCiAgICAgICAgUS52ZXJ0ZXhDb3VudCwKICAgICAgICB3LAogICAgICAgIE0sCiAgICAgICAgeSwKICAgICAgICBtLAogICAgICAgIEwKICAgICAgKSwgeSArIFEudmVydGV4Q291bnQgKiA0ID4gci5idWZmZXIuYnl0ZUxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIkRlcHRoIGluZGV4IGJ1ZmZlciBvdmVyZmxvdyIpOwogICAgICBjb25zdCBJID0gbmV3IFVpbnQzMkFycmF5KHIuYnVmZmVyLCB5LCBRLnZlcnRleENvdW50KSwgWSA9IG5ldyBVaW50MzJBcnJheShJLnNsaWNlKCkuYnVmZmVyKTsKICAgICAgc2VsZi5wb3N0TWVzc2FnZSh7IGRlcHRoSW5kZXg6IFkgfSwgW1kuYnVmZmVyXSk7CiAgICB9IGNhdGNoIHsKICAgICAgc2VsZi5wb3N0TWVzc2FnZSh7IGRlcHRoSW5kZXg6IG5ldyBVaW50MzJBcnJheSgwKSB9LCBbXSk7CiAgICB9CiAgICBoID0gITEsIEYgPSAhMTsKICB9Cn0sIGogPSAoKSA9PiB7CiAgRyB8fCAoRyA9ICEwLCBGICYmIG9BKCksIHNldFRpbWVvdXQoKCkgPT4gewogICAgRyA9ICExLCBqKCk7CiAgfSkpOwp9OwpzZWxmLm9ubWVzc2FnZSA9IChDKSA9PiB7CiAgQy5kYXRhLnNvcnREYXRhICYmIChRID8gKFEucG9zaXRpb25zLnNldChDLmRhdGEuc29ydERhdGEucG9zaXRpb25zKSwgUS50cmFuc2Zvcm1zLnNldChDLmRhdGEuc29ydERhdGEudHJhbnNmb3JtcyksIFEudHJhbnNmb3JtSW5kaWNlcy5zZXQoQy5kYXRhLnNvcnREYXRhLnRyYW5zZm9ybUluZGljZXMpLCBRLnZlcnRleENvdW50ID0gQy5kYXRhLnNvcnREYXRhLnZlcnRleENvdW50KSA6IFEgPSB7CiAgICBwb3NpdGlvbnM6IG5ldyBGbG9hdDMyQXJyYXkoQy5kYXRhLnNvcnREYXRhLnBvc2l0aW9ucyksCiAgICB0cmFuc2Zvcm1zOiBuZXcgRmxvYXQzMkFycmF5KEMuZGF0YS5zb3J0RGF0YS50cmFuc2Zvcm1zKSwKICAgIHRyYW5zZm9ybUluZGljZXM6IG5ldyBVaW50MzJBcnJheShDLmRhdGEuc29ydERhdGEudHJhbnNmb3JtSW5kaWNlcyksCiAgICB2ZXJ0ZXhDb3VudDogQy5kYXRhLnNvcnREYXRhLnZlcnRleENvdW50CiAgfSwgRiA9ICEwLCBaKCkpLCBDLmRhdGEudmlld1Byb2ogJiYgKEMuZGF0YS52aWV3UHJvai5ldmVyeSgocikgPT4gSy5pbmNsdWRlcyhyKSkgPT09ICExICYmIChLID0gQy5kYXRhLnZpZXdQcm9qLCBGID0gITApLCBqKCkpOwp9OwovLyMgc291cmNlTWFwcGluZ1VSTD1Tb3J0V29ya2VyLUJYQmJES3R1LmpzLm1hcAo=", Rt = (E) => Uint8Array.from(atob(E), (t) => t.charCodeAt(0)), ot = typeof self < "u" && self.Blob && new Blob(["URL.revokeObjectURL(import.meta.url);", Rt(Ut)], { type: "text/javascript;charset=utf-8" });
function Et(E) {
  let t;
  try {
    if (t = ot && (self.URL || self.webkitURL).createObjectURL(ot), !t) throw "";
    const n = new Worker(t, {
      type: "module",
      name: E?.name
    });
    return n.addEventListener("error", () => {
      (self.URL || self.webkitURL).revokeObjectURL(t);
    }), n;
  } catch {
    return new Worker(
      "data:text/javascript;base64," + Ut,
      {
        type: "module",
        name: E?.name
      }
    );
  }
}
class at {
  constructor(t, n) {
    this._scene = null, this._camera = null, this._started = !1, this._initialized = !1, this._renderer = t;
    const i = t.gl;
    this._program = i.createProgram(), this._passes = n || [];
    const e = i.createShader(i.VERTEX_SHADER);
    i.shaderSource(e, this._getVertexSource()), i.compileShader(e), i.getShaderParameter(e, i.COMPILE_STATUS) || console.error(i.getShaderInfoLog(e));
    const Q = i.createShader(i.FRAGMENT_SHADER);
    i.shaderSource(Q, this._getFragmentSource()), i.compileShader(Q), i.getShaderParameter(Q, i.COMPILE_STATUS) || console.error(i.getShaderInfoLog(Q)), i.attachShader(this.program, e), i.attachShader(this.program, Q), i.linkProgram(this.program), i.getProgramParameter(this.program, i.LINK_STATUS) || console.error(i.getProgramInfoLog(this.program)), this.resize = () => {
      i.useProgram(this._program), this._resize();
    }, this.initialize = () => {
      console.assert(!this._initialized, "ShaderProgram already initialized"), i.useProgram(this._program), this._initialize();
      for (const o of this.passes)
        o.initialize(this);
      this._initialized = !0, this._started = !0;
    }, this.render = (o, s) => {
      i.useProgram(this._program), (this._scene !== o || this._camera !== s) && (this.dispose(), this._scene = o, this._camera = s, this.initialize());
      for (const r of this.passes)
        r.render();
      this._render();
    }, this.dispose = () => {
      if (this._initialized) {
        i.useProgram(this._program);
        for (const o of this.passes)
          o.dispose();
        this._dispose(), this._scene = null, this._camera = null, this._initialized = !1;
      }
    };
  }
  get renderer() {
    return this._renderer;
  }
  get scene() {
    return this._scene;
  }
  get camera() {
    return this._camera;
  }
  get program() {
    return this._program;
  }
  get passes() {
    return this._passes;
  }
  get started() {
    return this._started;
  }
}
const Ft = "YXN5bmMgZnVuY3Rpb24gUUEoQyA9IHt9KSB7CiAgdmFyIGUsIEEgPSBDLCBjID0gaW1wb3J0Lm1ldGEudXJsLCBIID0gIiIsIHc7CiAgewogICAgdHJ5IHsKICAgICAgSCA9IG5ldyBVUkwoIi4iLCBjKS5ocmVmOwogICAgfSBjYXRjaCB7CiAgICB9CiAgICB3ID0gKEkpID0+IHsKICAgICAgdmFyIEIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTsKICAgICAgcmV0dXJuIEIub3BlbigiR0VUIiwgSSwgITEpLCBCLnJlc3BvbnNlVHlwZSA9ICJhcnJheWJ1ZmZlciIsIEIuc2VuZChudWxsKSwgbmV3IFVpbnQ4QXJyYXkoQi5yZXNwb25zZSk7CiAgICB9OwogIH0KICBjb25zb2xlLmxvZy5iaW5kKGNvbnNvbGUpLCBjb25zb2xlLmVycm9yLmJpbmQoY29uc29sZSk7CiAgdmFyIGgsIGwsIHksIE4sIGkgPSAhMTsKICBmdW5jdGlvbiBmKCkgewogICAgdmFyIEkgPSB5LmJ1ZmZlcjsKICAgIEEuSEVBUFU4ID0gTiA9IG5ldyBVaW50OEFycmF5KEkpLCBBLkhFQVBVMzIgPSBuZXcgVWludDMyQXJyYXkoSSksIEEuSEVBUEYzMiA9IG5ldyBGbG9hdDMyQXJyYXkoSSksIG5ldyBCaWdJbnQ2NEFycmF5KEkpLCBuZXcgQmlnVWludDY0QXJyYXkoSSk7CiAgfQogIGZ1bmN0aW9uIFooKSB7CiAgICBpZiAoQS5wcmVSdW4pCiAgICAgIGZvciAodHlwZW9mIEEucHJlUnVuID09ICJmdW5jdGlvbiIgJiYgKEEucHJlUnVuID0gW0EucHJlUnVuXSk7IEEucHJlUnVuLmxlbmd0aDsgKQogICAgICAgIHooQS5wcmVSdW4uc2hpZnQoKSk7CiAgICBtKHApOwogIH0KICBmdW5jdGlvbiBXKCkgewogICAgaSA9ICEwLCBELmMoKTsKICB9CiAgZnVuY3Rpb24gdigpIHsKICAgIGlmIChBLnBvc3RSdW4pCiAgICAgIGZvciAodHlwZW9mIEEucG9zdFJ1biA9PSAiZnVuY3Rpb24iICYmIChBLnBvc3RSdW4gPSBbQS5wb3N0UnVuXSk7IEEucG9zdFJ1bi5sZW5ndGg7ICkKICAgICAgICBfKEEucG9zdFJ1bi5zaGlmdCgpKTsKICAgIG0ocSk7CiAgfQogIHZhciBrOwogIGZ1bmN0aW9uIHgoKSB7CiAgICByZXR1cm4gVigiQUdGemJRRUFBQUFCSmdaZ0FYOEJmMkFDZlgwQmYyQUJmUUYvWUFGL0FHQUxmMzkvZjM5L2YzOS9mMzhBWUFBQUFnY0JBV0VCWVFBQUF3Z0hBQUVDQXdBRUJRVUhBUUdDQW9DQUFnWUlBWDhCUVlDTUJBc0hGUVVCWWdJQUFXTUFCd0ZrQUFZQlpRQUZBV1lBQkF3QkFRcVpRQWRVQWdGL0FYNENRRUdBQ0NnQ0FDSUJyU0FBclVJSGZFTDQvLy8vSDROOElnSkMvLy8vL3c5WUJFQWdBcWNpQUQ4QVFSQjBUUTBCSUFBUUFBMEJDMEdFQ0VFd05nSUFRWDhQQzBHQUNDQUFOZ0lBSUFFTERnQWdBQkFESUFFUUEwRVFkSElMY2dFRWZ5QUF2Q0lFUWYvLy93TnhJUUVDUUNBRVFSZDJRZjhCY1NJQ1JRMEFJQUpCOEFCTkJFQWdBVUdBZ0lBRWNrSHhBQ0FDYTNZaEFRd0JDeUFDUVkwQlN3UkFRWUQ0QVNFRFFRQWhBUXdCQ3lBQ1FRcDBRWUNBQjJzaEF3c2dBeUFFUVJCMlFZQ0FBbkZ5SUFGQkRYWnlDOThMQVFoL0FrQWdBRVVOQUNBQVFRaHJJZ01nQUVFRWF5Z0NBQ0lDUVhoeElnQnFJUVVDUUNBQ1FRRnhEUUFnQWtFQ2NVVU5BU0FESUFNb0FnQWlCR3NpQTBHWUNDZ0NBRWtOQVNBQUlBUnFJUUFDUUFKQUFrQkJuQWdvQWdBZ0EwY0VRQ0FES0FJTUlRRWdCRUgvQVUwRVFDQUJJQU1vQWdnaUFrY05Ba0dJQ0VHSUNDZ0NBRUYrSUFSQkEzWjNjVFlDQUF3RkN5QURLQUlZSVFjZ0FTQURSd1JBSUFNb0FnZ2lBaUFCTmdJTUlBRWdBallDQ0F3RUN5QURLQUlVSWdJRWZ5QURRUlJxQlNBREtBSVFJZ0pGRFFNZ0EwRVFhZ3NoQkFOQUlBUWhCaUFDSWdGQkZHb2hCQ0FCS0FJVUlnSU5BQ0FCUVJCcUlRUWdBU2dDRUNJQ0RRQUxJQVpCQURZQ0FBd0RDeUFGS0FJRUlnSkJBM0ZCQTBjTkEwR1FDQ0FBTmdJQUlBVWdBa0YrY1RZQ0JDQURJQUJCQVhJMkFnUWdCU0FBTmdJQUR3c2dBaUFCTmdJTUlBRWdBallDQ0F3Q0MwRUFJUUVMSUFkRkRRQUNRQ0FES0FJY0lnUkJBblFpQWlnQ3VBb2dBMFlFUUNBQ1FiZ0thaUFCTmdJQUlBRU5BVUdNQ0VHTUNDZ0NBRUYrSUFSM2NUWUNBQXdDQ3dKQUlBTWdCeWdDRUVZRVFDQUhJQUUyQWhBTUFRc2dCeUFCTmdJVUN5QUJSUTBCQ3lBQklBYzJBaGdnQXlnQ0VDSUNCRUFnQVNBQ05nSVFJQUlnQVRZQ0dBc2dBeWdDRkNJQ1JRMEFJQUVnQWpZQ0ZDQUNJQUUyQWhnTElBTWdCVThOQUNBRktBSUVJZ1JCQVhGRkRRQUNRQUpBQWtBQ1FDQUVRUUp4UlFSQVFhQUlLQUlBSUFWR0JFQkJvQWdnQXpZQ0FFR1VDRUdVQ0NnQ0FDQUFhaUlBTmdJQUlBTWdBRUVCY2pZQ0JDQURRWndJS0FJQVJ3MEdRWkFJUVFBMkFnQkJuQWhCQURZQ0FBOExRWndJS0FJQUlnY2dCVVlFUUVHY0NDQUROZ0lBUVpBSVFaQUlLQUlBSUFCcUlnQTJBZ0FnQXlBQVFRRnlOZ0lFSUFBZ0Eyb2dBRFlDQUE4TElBUkJlSEVnQUdvaEFDQUZLQUlNSVFFZ0JFSC9BVTBFUUNBRktBSUlJZ0lnQVVZRVFFR0lDRUdJQ0NnQ0FFRitJQVJCQTNaM2NUWUNBQXdGQ3lBQ0lBRTJBZ3dnQVNBQ05nSUlEQVFMSUFVb0FoZ2hDQ0FCSUFWSEJFQWdCU2dDQ0NJQ0lBRTJBZ3dnQVNBQ05nSUlEQU1MSUFVb0FoUWlBZ1IvSUFWQkZHb0ZJQVVvQWhBaUFrVU5BaUFGUVJCcUN5RUVBMEFnQkNFR0lBSWlBVUVVYWlFRUlBRW9BaFFpQWcwQUlBRkJFR29oQkNBQktBSVFJZ0lOQUFzZ0JrRUFOZ0lBREFJTElBVWdCRUYrY1RZQ0JDQURJQUJCQVhJMkFnUWdBQ0FEYWlBQU5nSUFEQU1MUVFBaEFRc2dDRVVOQUFKQUlBVW9BaHdpQkVFQ2RDSUNLQUs0Q2lBRlJnUkFJQUpCdUFwcUlBRTJBZ0FnQVEwQlFZd0lRWXdJS0FJQVFYNGdCSGR4TmdJQURBSUxBa0FnQlNBSUtBSVFSZ1JBSUFnZ0FUWUNFQXdCQ3lBSUlBRTJBaFFMSUFGRkRRRUxJQUVnQ0RZQ0dDQUZLQUlRSWdJRVFDQUJJQUkyQWhBZ0FpQUJOZ0lZQ3lBRktBSVVJZ0pGRFFBZ0FTQUNOZ0lVSUFJZ0FUWUNHQXNnQXlBQVFRRnlOZ0lFSUFBZ0Eyb2dBRFlDQUNBRElBZEhEUUJCa0FnZ0FEWUNBQThMSUFCQi93Rk5CRUFnQUVINEFYRkJzQWhxSVFJQ2YwR0lDQ2dDQUNJRVFRRWdBRUVEZG5RaUFIRkZCRUJCaUFnZ0FDQUVjallDQUNBQ0RBRUxJQUlvQWdnTElRQWdBaUFETmdJSUlBQWdBellDRENBRElBSTJBZ3dnQXlBQU5nSUlEd3RCSHlFQklBQkIvLy8vQjAwRVFDQUFRU1lnQUVFSWRtY2lBbXQyUVFGeElBSkJBWFJyUVQ1cUlRRUxJQU1nQVRZQ0hDQURRZ0EzQWhBZ0FVRUNkRUc0Q21vaEJBSi9Ba0FDZjBHTUNDZ0NBQ0lHUVFFZ0FYUWlBbkZGQkVCQmpBZ2dBaUFHY2pZQ0FDQUVJQU0yQWdCQkdDRUJRUWdNQVFzZ0FFRVpJQUZCQVhaclFRQWdBVUVmUnh0MElRRWdCQ2dDQUNFRUEwQWdCQ0lDS0FJRVFYaHhJQUJHRFFJZ0FVRWRkaUVFSUFGQkFYUWhBU0FDSUFSQkJIRnFJZ1lvQWhBaUJBMEFDeUFHSUFNMkFoQkJHQ0VCSUFJaEJFRUlDeUVBSUFNaUFnd0JDeUFDS0FJSUlnUWdBellDRENBQ0lBTTJBZ2hCR0NFQVFRZ2hBVUVBQ3lFR0lBRWdBMm9nQkRZQ0FDQURJQUkyQWd3Z0FDQURhaUFHTmdJQVFhZ0lRYWdJS0FJQVFRRnJJZ0JCZnlBQUd6WUNBQXNMdUNjQkMzOGpBRUVRYXlJS0pBQUNRQUpBQWtBQ1FBSkFBa0FDUUFKQUFrQUNRQ0FBUWZRQlRRUkFRWWdJS0FJQUlnUkJFQ0FBUVF0cVFmZ0RjU0FBUVF0Skd5SUdRUU4ySWdCMklnRkJBM0VFUUFKQUlBRkJmM05CQVhFZ0FHb2lBMEVEZENJQlFiQUlhaUlBSUFFb0FyZ0lJZ0lvQWdnaUJVWUVRRUdJQ0NBRVFYNGdBM2R4TmdJQURBRUxJQVVnQURZQ0RDQUFJQVUyQWdnTElBSkJDR29oQUNBQ0lBRkJBM0kyQWdRZ0FTQUNhaUlCSUFFb0FnUkJBWEkyQWdRTUN3c2dCa0dRQ0NnQ0FDSUlUUTBCSUFFRVFBSkFRUUlnQUhRaUFrRUFJQUpyY2lBQklBQjBjV2dpQTBFRGRDSUJRYkFJYWlJQ0lBRW9BcmdJSWdBb0FnZ2lCVVlFUUVHSUNDQUVRWDRnQTNkeElnUTJBZ0FNQVFzZ0JTQUNOZ0lNSUFJZ0JUWUNDQXNnQUNBR1FRTnlOZ0lFSUFBZ0Jtb2lCeUFCSUFacklnVkJBWEkyQWdRZ0FDQUJhaUFGTmdJQUlBZ0VRQ0FJUVhoeFFiQUlhaUVCUVp3SUtBSUFJUUlDZnlBRVFRRWdDRUVEZG5RaUEzRkZCRUJCaUFnZ0F5QUVjallDQUNBQkRBRUxJQUVvQWdnTElRTWdBU0FDTmdJSUlBTWdBallDRENBQ0lBRTJBZ3dnQWlBRE5nSUlDeUFBUVFocUlRQkJuQWdnQnpZQ0FFR1FDQ0FGTmdJQURBc0xRWXdJS0FJQUlndEZEUUVnQzJoQkFuUW9BcmdLSWdJb0FnUkJlSEVnQm1zaEF5QUNJUUVEUUFKQUlBRW9BaEFpQUVVRVFDQUJLQUlVSWdCRkRRRUxJQUFvQWdSQmVIRWdCbXNpQVNBRElBRWdBMGtpQVJzaEF5QUFJQUlnQVJzaEFpQUFJUUVNQVFzTElBSW9BaGdoQ1NBQ0lBSW9BZ3dpQUVjRVFDQUNLQUlJSWdFZ0FEWUNEQ0FBSUFFMkFnZ01DZ3NnQWlnQ0ZDSUJCSDhnQWtFVWFnVWdBaWdDRUNJQlJRMERJQUpCRUdvTElRVURRQ0FGSVFjZ0FTSUFRUlJxSVFVZ0FDZ0NGQ0lCRFFBZ0FFRVFhaUVGSUFBb0FoQWlBUTBBQ3lBSFFRQTJBZ0FNQ1F0QmZ5RUdJQUJCdjM5TERRQWdBRUVMYWlJQlFYaHhJUVpCakFnb0FnQWlCMFVOQUVFZklRaEJBQ0FHYXlFRElBQkI5UC8vQjAwRVFDQUdRU1lnQVVFSWRtY2lBR3QyUVFGeElBQkJBWFJyUVQ1cUlRZ0xBa0FDUUFKQUlBaEJBblFvQXJnS0lnRkZCRUJCQUNFQURBRUxRUUFoQUNBR1FSa2dDRUVCZG10QkFDQUlRUjlIRzNRaEFnTkFBa0FnQVNnQ0JFRjRjU0FHYXlJRUlBTlBEUUFnQVNFRklBUWlBdzBBUVFBaEF5QUJJUUFNQXdzZ0FDQUJLQUlVSWdRZ0JDQUJJQUpCSFhaQkJIRnFLQUlRSWdGR0d5QUFJQVFiSVFBZ0FrRUJkQ0VDSUFFTkFBc0xJQUFnQlhKRkJFQkJBQ0VGUVFJZ0NIUWlBRUVBSUFCcmNpQUhjU0lBUlEwRElBQm9RUUowS0FLNENpRUFDeUFBUlEwQkN3TkFJQUFvQWdSQmVIRWdCbXNpQWlBRFNTRUJJQUlnQXlBQkd5RURJQUFnQlNBQkd5RUZJQUFvQWhBaUFRUi9JQUVGSUFBb0FoUUxJZ0FOQUFzTElBVkZEUUFnQTBHUUNDZ0NBQ0FHYTA4TkFDQUZLQUlZSVFnZ0JTQUZLQUlNSWdCSEJFQWdCU2dDQ0NJQklBQTJBZ3dnQUNBQk5nSUlEQWdMSUFVb0FoUWlBUVIvSUFWQkZHb0ZJQVVvQWhBaUFVVU5BeUFGUVJCcUN5RUNBMEFnQWlFRUlBRWlBRUVVYWlFQ0lBQW9BaFFpQVEwQUlBQkJFR29oQWlBQUtBSVFJZ0VOQUFzZ0JFRUFOZ0lBREFjTElBWkJrQWdvQWdBaUJVMEVRRUdjQ0NnQ0FDRUFBa0FnQlNBR2F5SUJRUkJQQkVBZ0FDQUdhaUlDSUFGQkFYSTJBZ1FnQUNBRmFpQUJOZ0lBSUFBZ0JrRURjallDQkF3QkN5QUFJQVZCQTNJMkFnUWdBQ0FGYWlJQklBRW9BZ1JCQVhJMkFnUkJBQ0VDUVFBaEFRdEJrQWdnQVRZQ0FFR2NDQ0FDTmdJQUlBQkJDR29oQUF3SkN5QUdRWlFJS0FJQUlnSkpCRUJCbEFnZ0FpQUdheUlCTmdJQVFhQUlRYUFJS0FJQUlnQWdCbW9pQWpZQ0FDQUNJQUZCQVhJMkFnUWdBQ0FHUVFOeU5nSUVJQUJCQ0dvaEFBd0pDMEVBSVFBZ0JrRXZhaUlEQW45QjRBc29BZ0FFUUVIb0N5Z0NBQXdCQzBIc0MwSi9Od0lBUWVRTFFvQ2dnSUNBZ0FRM0FnQkI0QXNnQ2tFTWFrRndjVUhZcXRXcUJYTTJBZ0JCOUF0QkFEWUNBRUhFQzBFQU5nSUFRWUFnQ3lJQmFpSUVRUUFnQVdzaUIzRWlBU0FHVFEwSVFjQUxLQUlBSWdVRVFFRzRDeWdDQUNJSUlBRnFJZ2tnQ0UwZ0JTQUpTWElOQ1FzQ1FFSEVDeTBBQUVFRWNVVUVRQUpBQWtBQ1FBSkFRYUFJS0FJQUlnVUVRRUhJQ3lFQUEwQWdBQ2dDQUNJSUlBVk5CRUFnQlNBSUlBQW9BZ1JxU1EwREN5QUFLQUlJSWdBTkFBc0xRUUFRQVNJQ1FYOUdEUU1nQVNFRVFlUUxLQUlBSWdCQkFXc2lCU0FDY1FSQUlBRWdBbXNnQWlBRmFrRUFJQUJyY1dvaEJBc2dCQ0FHVFEwRFFjQUxLQUlBSWdBRVFFRzRDeWdDQUNJRklBUnFJZ2NnQlUwZ0FDQUhTWElOQkFzZ0JCQUJJZ0FnQWtjTkFRd0ZDeUFFSUFKcklBZHhJZ1FRQVNJQ0lBQW9BZ0FnQUNnQ0JHcEdEUUVnQWlFQUN5QUFRWDlHRFFFZ0JrRXdhaUFFVFFSQUlBQWhBZ3dFQzBIb0N5Z0NBQ0lDSUFNZ0JHdHFRUUFnQW10eElnSVFBVUYvUmcwQklBSWdCR29oQkNBQUlRSU1Bd3NnQWtGL1J3MENDMEhFQzBIRUN5Z0NBRUVFY2pZQ0FBc2dBUkFCSWdKQmYwWkJBQkFCSWdCQmYwWnlJQUFnQWsxeURRVWdBQ0FDYXlJRUlBWkJLR3BORFFVTFFiZ0xRYmdMS0FJQUlBUnFJZ0EyQWdCQnZBc29BZ0FnQUVrRVFFRzhDeUFBTmdJQUN3SkFRYUFJS0FJQUlnTUVRRUhJQ3lFQUEwQWdBaUFBS0FJQUlnRWdBQ2dDQkNJRmFrWU5BaUFBS0FJSUlnQU5BQXNNQkF0Qm1BZ29BZ0FpQUVFQUlBQWdBazBiUlFSQVFaZ0lJQUkyQWdBTFFRQWhBRUhNQ3lBRU5nSUFRY2dMSUFJMkFnQkJxQWhCZnpZQ0FFR3NDRUhnQ3lnQ0FEWUNBRUhVQzBFQU5nSUFBMEFnQUVFRGRDSUJJQUZCc0FocUlnVTJBcmdJSUFFZ0JUWUN2QWdnQUVFQmFpSUFRU0JIRFFBTFFaUUlJQVJCS0dzaUFFRjRJQUpyUVFkeElnRnJJZ1UyQWdCQm9BZ2dBU0FDYWlJQk5nSUFJQUVnQlVFQmNqWUNCQ0FBSUFKcVFTZzJBZ1JCcEFoQjhBc29BZ0EyQWdBTUJBc2dBaUFEVFNBQklBTkxjZzBDSUFBb0FneEJDSEVOQWlBQUlBUWdCV28yQWdSQm9BZ2dBMEY0SUFOclFRZHhJZ0JxSWdFMkFnQkJsQWhCbEFnb0FnQWdCR29pQWlBQWF5SUFOZ0lBSUFFZ0FFRUJjallDQkNBQ0lBTnFRU2cyQWdSQnBBaEI4QXNvQWdBMkFnQU1Bd3RCQUNFQURBWUxRUUFoQUF3RUMwR1lDQ2dDQUNBQ1N3UkFRWmdJSUFJMkFnQUxJQUlnQkdvaEJVSElDeUVBQWtBRFFDQUZJQUFvQWdBaUFVY0VRQ0FBS0FJSUlnQU5BUXdDQ3dzZ0FDMEFERUVJY1VVTkF3dEJ5QXNoQUFOQUFrQWdBQ2dDQUNJQklBTk5CRUFnQXlBQklBQW9BZ1JxSWdWSkRRRUxJQUFvQWdnaEFBd0JDd3RCbEFnZ0JFRW9heUlBUVhnZ0FtdEJCM0VpQVdzaUJ6WUNBRUdnQ0NBQklBSnFJZ0UyQWdBZ0FTQUhRUUZ5TmdJRUlBQWdBbXBCS0RZQ0JFR2tDRUh3Q3lnQ0FEWUNBQ0FESUFWQkp5QUZhMEVIY1dwQkwyc2lBQ0FBSUFOQkVHcEpHeUlCUVJzMkFnUWdBVUhRQ3lrQ0FEY0NFQ0FCUWNnTEtRSUFOd0lJUWRBTElBRkJDR28yQWdCQnpBc2dCRFlDQUVISUN5QUNOZ0lBUWRRTFFRQTJBZ0FnQVVFWWFpRUFBMEFnQUVFSE5nSUVJQUJCQ0dvZ0FFRUVhaUVBSUFWSkRRQUxJQUVnQTBZTkFDQUJJQUVvQWdSQmZuRTJBZ1FnQXlBQklBTnJJZ0pCQVhJMkFnUWdBU0FDTmdJQUFuOGdBa0gvQVUwRVFDQUNRZmdCY1VHd0NHb2hBQUovUVlnSUtBSUFJZ0ZCQVNBQ1FRTjJkQ0lDY1VVRVFFR0lDQ0FCSUFKeU5nSUFJQUFNQVFzZ0FDZ0NDQXNoQVNBQUlBTTJBZ2dnQVNBRE5nSU1RUXdoQWtFSURBRUxRUjhoQUNBQ1FmLy8vd2ROQkVBZ0FrRW1JQUpCQ0habklnQnJka0VCY1NBQVFRRjBhMEUrYWlFQUN5QURJQUEyQWh3Z0EwSUFOd0lRSUFCQkFuUkJ1QXBxSVFFQ1FBSkFRWXdJS0FJQUlnVkJBU0FBZENJRWNVVUVRRUdNQ0NBRUlBVnlOZ0lBSUFFZ0F6WUNBQXdCQ3lBQ1FSa2dBRUVCZG10QkFDQUFRUjlIRzNRaEFDQUJLQUlBSVFVRFFDQUZJZ0VvQWdSQmVIRWdBa1lOQWlBQVFSMTJJUVVnQUVFQmRDRUFJQUVnQlVFRWNXb2lCQ2dDRUNJRkRRQUxJQVFnQXpZQ0VBc2dBeUFCTmdJWVFRZ2hBaUFESWdFaEFFRU1EQUVMSUFFb0FnZ2lBQ0FETmdJTUlBRWdBellDQ0NBRElBQTJBZ2hCQUNFQVFSZ2hBa0VNQ3lBRGFpQUJOZ0lBSUFJZ0Eyb2dBRFlDQUF0QmxBZ29BZ0FpQUNBR1RRMEFRWlFJSUFBZ0Jtc2lBVFlDQUVHZ0NFR2dDQ2dDQUNJQUlBWnFJZ0kyQWdBZ0FpQUJRUUZ5TmdJRUlBQWdCa0VEY2pZQ0JDQUFRUWhxSVFBTUJBdEJoQWhCTURZQ0FFRUFJUUFNQXdzZ0FDQUNOZ0lBSUFBZ0FDZ0NCQ0FFYWpZQ0JDQUNRWGdnQW10QkIzRnFJZ2dnQmtFRGNqWUNCQ0FCUVhnZ0FXdEJCM0ZxSWdRZ0JpQUlhaUlEYXlFSEFrQkJvQWdvQWdBZ0JFWUVRRUdnQ0NBRE5nSUFRWlFJUVpRSUtBSUFJQWRxSWdBMkFnQWdBeUFBUVFGeU5nSUVEQUVMUVp3SUtBSUFJQVJHQkVCQm5BZ2dBellDQUVHUUNFR1FDQ2dDQUNBSGFpSUFOZ0lBSUFNZ0FFRUJjallDQkNBQUlBTnFJQUEyQWdBTUFRc2dCQ2dDQkNJQVFRTnhRUUZHQkVBZ0FFRjRjU0VKSUFRb0Fnd2hBZ0pBSUFCQi93Rk5CRUFnQkNnQ0NDSUJJQUpHQkVCQmlBaEJpQWdvQWdCQmZpQUFRUU4yZDNFMkFnQU1BZ3NnQVNBQ05nSU1JQUlnQVRZQ0NBd0JDeUFFS0FJWUlRWUNRQ0FDSUFSSEJFQWdCQ2dDQ0NJQUlBSTJBZ3dnQWlBQU5nSUlEQUVMQWtBZ0JDZ0NGQ0lBQkg4Z0JFRVVhZ1VnQkNnQ0VDSUFSUTBCSUFSQkVHb0xJUUVEUUNBQklRVWdBQ0lDUVJScUlRRWdBQ2dDRkNJQURRQWdBa0VRYWlFQklBSW9BaEFpQUEwQUN5QUZRUUEyQWdBTUFRdEJBQ0VDQ3lBR1JRMEFBa0FnQkNnQ0hDSUFRUUowSWdFb0FyZ0tJQVJHQkVBZ0FVRzRDbW9nQWpZQ0FDQUNEUUZCakFoQmpBZ29BZ0JCZmlBQWQzRTJBZ0FNQWdzQ1FDQUVJQVlvQWhCR0JFQWdCaUFDTmdJUURBRUxJQVlnQWpZQ0ZBc2dBa1VOQVFzZ0FpQUdOZ0lZSUFRb0FoQWlBQVJBSUFJZ0FEWUNFQ0FBSUFJMkFoZ0xJQVFvQWhRaUFFVU5BQ0FDSUFBMkFoUWdBQ0FDTmdJWUN5QUhJQWxxSVFjZ0JDQUphaUlFS0FJRUlRQUxJQVFnQUVGK2NUWUNCQ0FESUFkQkFYSTJBZ1FnQXlBSGFpQUhOZ0lBSUFkQi93Rk5CRUFnQjBINEFYRkJzQWhxSVFBQ2YwR0lDQ2dDQUNJQlFRRWdCMEVEZG5RaUFuRkZCRUJCaUFnZ0FTQUNjallDQUNBQURBRUxJQUFvQWdnTElRRWdBQ0FETmdJSUlBRWdBellDRENBRElBQTJBZ3dnQXlBQk5nSUlEQUVMUVI4aEFpQUhRZi8vL3dkTkJFQWdCMEVtSUFkQkNIWm5JZ0JyZGtFQmNTQUFRUUYwYTBFK2FpRUNDeUFESUFJMkFod2dBMElBTndJUUlBSkJBblJCdUFwcUlRQUNRQUpBUVl3SUtBSUFJZ0ZCQVNBQ2RDSUZjVVVFUUVHTUNDQUJJQVZ5TmdJQUlBQWdBellDQUF3QkN5QUhRUmtnQWtFQmRtdEJBQ0FDUVI5SEczUWhBaUFBS0FJQUlRRURRQ0FCSWdBb0FnUkJlSEVnQjBZTkFpQUNRUjEySVFFZ0FrRUJkQ0VDSUFBZ0FVRUVjV29pQlNnQ0VDSUJEUUFMSUFVZ0F6WUNFQXNnQXlBQU5nSVlJQU1nQXpZQ0RDQURJQU0yQWdnTUFRc2dBQ2dDQ0NJQklBTTJBZ3dnQUNBRE5nSUlJQU5CQURZQ0dDQURJQUEyQWd3Z0F5QUJOZ0lJQ3lBSVFRaHFJUUFNQWdzQ1FDQUlSUTBBQWtBZ0JTZ0NIQ0lCUVFKMElnSW9BcmdLSUFWR0JFQWdBa0c0Q21vZ0FEWUNBQ0FBRFFGQmpBZ2dCMEYrSUFGM2NTSUhOZ0lBREFJTEFrQWdCU0FJS0FJUVJnUkFJQWdnQURZQ0VBd0JDeUFJSUFBMkFoUUxJQUJGRFFFTElBQWdDRFlDR0NBRktBSVFJZ0VFUUNBQUlBRTJBaEFnQVNBQU5nSVlDeUFGS0FJVUlnRkZEUUFnQUNBQk5nSVVJQUVnQURZQ0dBc0NRQ0FEUVE5TkJFQWdCU0FESUFacUlnQkJBM0kyQWdRZ0FDQUZhaUlBSUFBb0FnUkJBWEkyQWdRTUFRc2dCU0FHUVFOeU5nSUVJQVVnQm1vaUJDQURRUUZ5TmdJRUlBTWdCR29nQXpZQ0FDQURRZjhCVFFSQUlBTkIrQUZ4UWJBSWFpRUFBbjlCaUFnb0FnQWlBVUVCSUFOQkEzWjBJZ0p4UlFSQVFZZ0lJQUVnQW5JMkFnQWdBQXdCQ3lBQUtBSUlDeUVCSUFBZ0JEWUNDQ0FCSUFRMkFnd2dCQ0FBTmdJTUlBUWdBVFlDQ0F3QkMwRWZJUUFnQTBILy8vOEhUUVJBSUFOQkppQURRUWgyWnlJQWEzWkJBWEVnQUVFQmRHdEJQbW9oQUFzZ0JDQUFOZ0ljSUFSQ0FEY0NFQ0FBUVFKMFFiZ0thaUVCQWtBQ1FDQUhRUUVnQUhRaUFuRkZCRUJCakFnZ0FpQUhjallDQUNBQklBUTJBZ0FnQkNBQk5nSVlEQUVMSUFOQkdTQUFRUUYyYTBFQUlBQkJIMGNiZENFQUlBRW9BZ0FoQVFOQUlBRWlBaWdDQkVGNGNTQURSZzBDSUFCQkhYWWhBU0FBUVFGMElRQWdBaUFCUVFSeGFpSUhLQUlRSWdFTkFBc2dCeUFFTmdJUUlBUWdBallDR0FzZ0JDQUVOZ0lNSUFRZ0JEWUNDQXdCQ3lBQ0tBSUlJZ0FnQkRZQ0RDQUNJQVEyQWdnZ0JFRUFOZ0lZSUFRZ0FqWUNEQ0FFSUFBMkFnZ0xJQVZCQ0dvaEFBd0JDd0pBSUFsRkRRQUNRQ0FDS0FJY0lnRkJBblFpQlNnQ3VBb2dBa1lFUUNBRlFiZ0thaUFBTmdJQUlBQU5BVUdNQ0NBTFFYNGdBWGR4TmdJQURBSUxBa0FnQWlBSktBSVFSZ1JBSUFrZ0FEWUNFQXdCQ3lBSklBQTJBaFFMSUFCRkRRRUxJQUFnQ1RZQ0dDQUNLQUlRSWdFRVFDQUFJQUUyQWhBZ0FTQUFOZ0lZQ3lBQ0tBSVVJZ0ZGRFFBZ0FDQUJOZ0lVSUFFZ0FEWUNHQXNDUUNBRFFROU5CRUFnQWlBRElBWnFJZ0JCQTNJMkFnUWdBQ0FDYWlJQUlBQW9BZ1JCQVhJMkFnUU1BUXNnQWlBR1FRTnlOZ0lFSUFJZ0Jtb2lCU0FEUVFGeU5nSUVJQU1nQldvZ0F6WUNBQ0FJQkVBZ0NFRjRjVUd3Q0dvaEFFR2NDQ2dDQUNFQkFuOUJBU0FJUVFOMmRDSUhJQVJ4UlFSQVFZZ0lJQVFnQjNJMkFnQWdBQXdCQ3lBQUtBSUlDeUVFSUFBZ0FUWUNDQ0FFSUFFMkFnd2dBU0FBTmdJTUlBRWdCRFlDQ0F0Qm5BZ2dCVFlDQUVHUUNDQUROZ0lBQ3lBQ1FRaHFJUUFMSUFwQkVHb2tBQ0FBQzZFTEFndC9DWDBqQUVHZ0FXc2lDeVFBSUF0Qk1HcEJBRUVrL0FzQUEwQWdBU0FPUndSQUlBSWdEa0VEYkNJTVFRSnFRUUowSWc5cUtnSUFJUmNnQWlBTVFRRnFRUUowSWhCcUtnSUFJUmdnQ0NBTVFRSjBJaEZxSUFJZ0VXb3FBZ0FpR1RnQ0FDQUlJQkJxSUJnNEFnQWdDQ0FQYWlBWE9BSUFJQWNnRGtFRmRHb2lEVUVBTmdJTUlBMGdGemdDQ0NBTklCZzRBZ1FnRFNBWk9BSUFBa0FnQUVVRVFDQUdJQTVxTFFBQVJRMEJDeUFOUVlDQWdBZzJBZ3dMSUEwZ0JTQU9RUUowSWd4QkFYSWlFbW90QUFCQkNIUWdCU0FNYWkwQUFISWdCU0FNUVFKeUloTnFMUUFBUVJCMGNpQUZJQXhCQTNJaURHb3RBQUJCR0hSeU5nSWNJQXNnQXlBU1FRSjBJaEpxS2dJQUloYzRBcEFCSUFzZ0F5QVRRUUowSWhOcUtnSUFJaGc0QXBRQklBc2dBeUFNUVFKMEloUnFLZ0lBSWhrNEFwZ0JJQXNnQXlBT1FRUjBJaFZxS2dJQWpDSWFPQUtjQVNBTFFlQUFhaUlNSUFzcUFwZ0JJaFpEQUFBQXdKUWdGcFFnQ3lvQ2xBRWlGa01BQUFEQWxDQVdsRU1BQUlBL2twSTRBZ0FnRENBTEtnS1FBU0lXSUJhU0lBc3FBcFFCbENBTEtnS1lBU0lXSUJhU0lBc3FBcHdCbEpNNEFnUWdEQ0FMS2dLUUFTSVdJQmFTSUFzcUFwZ0JsQ0FMS2dLVUFTSVdJQmFTSUFzcUFwd0JsSkk0QWdnZ0RDQUxLZ0tRQVNJV0lCYVNJQXNxQXBRQmxDQUxLZ0tZQVNJV0lCYVNJQXNxQXB3QmxKSTRBZ3dnRENBTEtnS1lBU0lXUXdBQUFNQ1VJQmFVSUFzcUFwQUJJaFpEQUFBQXdKUWdGcFJEQUFDQVA1S1NPQUlRSUF3Z0N5b0NsQUVpRmlBV2tpQUxLZ0tZQVpRZ0N5b0NrQUVpRmlBV2tpQUxLZ0tjQVpTVE9BSVVJQXdnQ3lvQ2tBRWlGaUFXa2lBTEtnS1lBWlFnQ3lvQ2xBRWlGaUFXa2lBTEtnS2NBWlNUT0FJWUlBd2dDeW9DbEFFaUZpQVdraUFMS2dLWUFaUWdDeW9Da0FFaUZpQVdraUFMS2dLY0FaU1NPQUljSUF3Z0N5b0NsQUVpRmtNQUFBREFsQ0FXbENBTEtnS1FBU0lXUXdBQUFNQ1VJQmFVUXdBQWdEK1NramdDSUNBSklCVnFJQmM0QWdBZ0NTQVNhaUFZT0FJQUlBa2dFMm9nR1RnQ0FDQUpJQlJxSUJvNEFnQWdDeUFFSUJGcUtnSUFJaGM0QWpBZ0N5QUVJQkJxS2dJQUloZzRBa0FnQ3lBRUlBOXFLZ0lBSWhrNEFsQWdDaUFSYWlBWE9BSUFJQW9nRUdvZ0dEZ0NBQ0FLSUE5cUlCazRBZ0FnQ3lBTUtnSVlJQXNxQWppVUlBd3FBZ0FnQ3lvQ01KUWdEQ29DRENBTEtnSTBsSktTT0FJQUlBc2dEQ29DSENBTEtnSTRsQ0FNS2dJRUlBc3FBakNVSUF3cUFoQWdDeW9DTkpTU2tqZ0NCQ0FMSUF3cUFpQWdDeW9DT0pRZ0RDb0NDQ0FMS2dJd2xDQU1LZ0lVSUFzcUFqU1VrcEk0QWdnZ0N5QU1LZ0lZSUFzcUFrU1VJQXdxQWdBZ0N5b0NQSlFnRENvQ0RDQUxLZ0pBbEpLU09BSU1JQXNnRENvQ0hDQUxLZ0pFbENBTUtnSUVJQXNxQWp5VUlBd3FBaEFnQ3lvQ1FKU1NramdDRUNBTElBd3FBaUFnQ3lvQ1JKUWdEQ29DQ0NBTEtnSThsQ0FNS2dJVUlBc3FBa0NVa3BJNEFoUWdDeUFNS2dJWUlBc3FBbENVSUF3cUFnQWdDeW9DU0pRZ0RDb0NEQ0FMS2dKTWxKS1NPQUlZSUFzZ0RDb0NIQ0FMS2dKUWxDQU1LZ0lFSUFzcUFraVVJQXdxQWhBZ0N5b0NUSlNTa2pnQ0hDQUxJQXdxQWlBZ0N5b0NVSlFnRENvQ0NDQUxLZ0pJbENBTUtnSVVJQXNxQWt5VWtwSTRBaUFnQ3lvQ0lDRVhJQXNxQWdnaEdDQUxLZ0lVSVJrZ0RTQUxLZ0lZSWhvZ0dwUWdDeW9DQUNJV0lCYVVJQXNxQWd3aUd5QWJsSktTUXdBQWdFQ1VJQm9nQ3lvQ0hDSWNsQ0FXSUFzcUFnUWlIWlFnR3lBTEtnSVFJaDZVa3BKREFBQ0FRSlFRQWpZQ0VDQU5JQm9nRjVRZ0ZpQVlsQ0FiSUJtVWtwSkRBQUNBUUpRZ0hDQWNsQ0FkSUIyVUlCNGdIcFNTa2tNQUFJQkFsQkFDTmdJVUlBMGdIQ0FYbENBZElCaVVJQjRnR1pTU2trTUFBSUJBbENBWElCZVVJQmdnR0pRZ0dTQVpsSktTUXdBQWdFQ1VFQUkyQWhnZ0RrRUJhaUVPREFFTEN5QUxRYUFCYWlRQUN3SUFDd3NKQVFCQmdRZ0xBZ1lCIik7CiAgfQogIGZ1bmN0aW9uIFAoSSkgewogICAgaWYgKEFycmF5QnVmZmVyLmlzVmlldyhJKSkKICAgICAgcmV0dXJuIEk7CiAgICBpZiAoSSA9PSBrICYmIGgpCiAgICAgIHJldHVybiBuZXcgVWludDhBcnJheShoKTsKICAgIGlmICh3KQogICAgICByZXR1cm4gdyhJKTsKICAgIHRocm93ICdzeW5jIGZldGNoaW5nIG9mIHRoZSB3YXNtIGZhaWxlZDogeW91IGNhbiBwcmVsb2FkIGl0IHRvIE1vZHVsZVsid2FzbUJpbmFyeSJdIG1hbnVhbGx5LCBvciBlbWNjLnB5IHdpbGwgZG8gdGhhdCBmb3IgeW91IHdoZW4gZ2VuZXJhdGluZyBIVE1MIChidXQgbm90IEpTKSc7CiAgfQogIGZ1bmN0aW9uIGooSSwgQikgewogICAgdmFyIEUsIFEgPSBQKEkpOwogICAgRSA9IG5ldyBXZWJBc3NlbWJseS5Nb2R1bGUoUSk7CiAgICB2YXIgbyA9IG5ldyBXZWJBc3NlbWJseS5JbnN0YW5jZShFLCBCKTsKICAgIHJldHVybiBbbywgRV07CiAgfQogIGZ1bmN0aW9uIFQoKSB7CiAgICByZXR1cm4geyBhOiBDQSB9OwogIH0KICBmdW5jdGlvbiBiKCkgewogICAgZnVuY3Rpb24gSShRLCBvKSB7CiAgICAgIHJldHVybiBEID0gUS5leHBvcnRzLCB5ID0gRC5iLCBmKCksIGdBKEQpLCBEOwogICAgfQogICAgdmFyIEIgPSBUKCk7CiAgICBpZiAoQS5pbnN0YW50aWF0ZVdhc20pCiAgICAgIHJldHVybiBuZXcgUHJvbWlzZSgoUSwgbykgPT4gewogICAgICAgIEEuaW5zdGFudGlhdGVXYXNtKEIsIChuLCBzKSA9PiB7CiAgICAgICAgICBRKEkobikpOwogICAgICAgIH0pOwogICAgICB9KTsKICAgIGsgPz89IHgoKTsKICAgIHZhciBFID0gaihrLCBCKTsKICAgIHJldHVybiBJKEVbMF0pOwogIH0KICBmb3IgKHZhciBtID0gKEkpID0+IHsKICAgIGZvciAoOyBJLmxlbmd0aCA+IDA7ICkKICAgICAgSS5zaGlmdCgpKEEpOwogIH0sIHEgPSBbXSwgXyA9IChJKSA9PiBxLnB1c2goSSksIHAgPSBbXSwgeiA9IChJKSA9PiBwLnB1c2goSSksIFYgPSAoSSkgPT4gewogICAgZm9yICh2YXIgQiwgRSwgUSA9IDAsIG8gPSAwLCBuID0gSS5sZW5ndGgsIHMgPSBuZXcgVWludDhBcnJheSgobiAqIDMgPj4gMikgLSAoSVtuIC0gMl0gPT0gIj0iKSAtIChJW24gLSAxXSA9PSAiPSIpKTsgUSA8IG47IFEgKz0gNCwgbyArPSAzKQogICAgICBCID0gcltJLmNoYXJDb2RlQXQoUSArIDEpXSwgRSA9IHJbSS5jaGFyQ29kZUF0KFEgKyAyKV0sIHNbb10gPSByW0kuY2hhckNvZGVBdChRKV0gPDwgMiB8IEIgPj4gNCwgc1tvICsgMV0gPSBCIDw8IDQgfCBFID4+IDIsIHNbbyArIDJdID0gRSA8PCA2IHwgcltJLmNoYXJDb2RlQXQoUSArIDMpXTsKICAgIHJldHVybiBzOwogIH0sIE8gPSAoKSA9PiAyMTQ3NDgzNjQ4LCAkID0gKEksIEIpID0+IE1hdGguY2VpbChJIC8gQikgKiBCLCBBQSA9IChJKSA9PiB7CiAgICB2YXIgQiA9IHkuYnVmZmVyLmJ5dGVMZW5ndGgsIEUgPSAoSSAtIEIgKyA2NTUzNSkgLyA2NTUzNiB8IDA7CiAgICB0cnkgewogICAgICByZXR1cm4geS5ncm93KEUpLCBmKCksIDE7CiAgICB9IGNhdGNoIHsKICAgIH0KICB9LCBJQSA9IChJKSA9PiB7CiAgICB2YXIgQiA9IE4ubGVuZ3RoOwogICAgSSA+Pj49IDA7CiAgICB2YXIgRSA9IE8oKTsKICAgIGlmIChJID4gRSkKICAgICAgcmV0dXJuICExOwogICAgZm9yICh2YXIgUSA9IDE7IFEgPD0gNDsgUSAqPSAyKSB7CiAgICAgIHZhciBvID0gQiAqICgxICsgMC4yIC8gUSk7CiAgICAgIG8gPSBNYXRoLm1pbihvLCBJICsgMTAwNjYzMjk2KTsKICAgICAgdmFyIG4gPSBNYXRoLm1pbihFLCAkKE1hdGgubWF4KEksIG8pLCA2NTUzNikpLCBzID0gQUEobik7CiAgICAgIGlmIChzKQogICAgICAgIHJldHVybiAhMDsKICAgIH0KICAgIHJldHVybiAhMTsKICB9LCByID0gbmV3IFVpbnQ4QXJyYXkoMTIzKSwgYSA9IDI1OyBhID49IDA7IC0tYSkKICAgIHJbNDggKyBhXSA9IDUyICsgYSwgcls2NSArIGFdID0gYSwgcls5NyArIGFdID0gMjYgKyBhOwogIGlmIChyWzQzXSA9IDYyLCByWzQ3XSA9IDYzLCBBLm5vRXhpdFJ1bnRpbWUgJiYgQS5ub0V4aXRSdW50aW1lLCBBLnByaW50ICYmIEEucHJpbnQsIEEucHJpbnRFcnIgJiYgQS5wcmludEVyciwgQS53YXNtQmluYXJ5ICYmIChoID0gQS53YXNtQmluYXJ5KSwgQS5hcmd1bWVudHMgJiYgQS5hcmd1bWVudHMsIEEudGhpc1Byb2dyYW0gJiYgQS50aGlzUHJvZ3JhbSwgQS5wcmVJbml0KQogICAgZm9yICh0eXBlb2YgQS5wcmVJbml0ID09ICJmdW5jdGlvbiIgJiYgKEEucHJlSW5pdCA9IFtBLnByZUluaXRdKTsgQS5wcmVJbml0Lmxlbmd0aCA+IDA7ICkKICAgICAgQS5wcmVJbml0LnNoaWZ0KCkoKTsKICBmdW5jdGlvbiBnQShJKSB7CiAgICBBLl9wYWNrID0gSS5kLCBBLl9tYWxsb2MgPSBJLmUsIEEuX2ZyZWUgPSBJLmY7CiAgfQogIHZhciBDQSA9IHsgYTogSUEgfTsKICBmdW5jdGlvbiBCQSgpIHsKICAgIFooKTsKICAgIGZ1bmN0aW9uIEkoKSB7CiAgICAgIEEuY2FsbGVkUnVuID0gITAsIFcoKSwgbD8uKEEpLCBBLm9uUnVudGltZUluaXRpYWxpemVkPy4oKSwgdigpOwogICAgfQogICAgQS5zZXRTdGF0dXMgPyAoQS5zZXRTdGF0dXMoIlJ1bm5pbmcuLi4iKSwgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHNldFRpbWVvdXQoKCkgPT4gQS5zZXRTdGF0dXMoIiIpLCAxKSwgSSgpOwogICAgfSwgMSkpIDogSSgpOwogIH0KICB2YXIgRDsKICByZXR1cm4gRCA9IGIoKSwgQkEoKSwgaSA/IGUgPSBBIDogZSA9IG5ldyBQcm9taXNlKChJLCBCKSA9PiB7CiAgICBsID0gSTsKICB9KSwgZTsKfQpsZXQgZzsKYXN5bmMgZnVuY3Rpb24gRUEoKSB7CiAgaWYgKGcgPSBhd2FpdCBRQSgpLCAhZyB8fCAhZy5IRUFQRjMyIHx8ICFnLl9wYWNrKQogICAgdGhyb3cgbmV3IEVycm9yKCJXQVNNIG1vZHVsZSBmYWlsZWQgdG8gaW5pdGlhbGl6ZSBwcm9wZXJseSIpOwp9CmxldCB0ID0gMDsKY29uc3QgRiA9IG5ldyBBcnJheSgpOwpsZXQgTCA9ICExLCBKID0gITEsIFksIEcsIFIsIFUsIEssIFMsIHUsIE0sIGQ7CmNvbnN0IGlBID0gYXN5bmMgKEMpID0+IHsKICBmb3IgKDsgSjsgKQogICAgYXdhaXQgbmV3IFByb21pc2UoKGYpID0+IHNldFRpbWVvdXQoZiwgMCkpOwogIGcgfHwgKEogPSAhMCwgYXdhaXQgRUEoKSwgSiA9ICExKTsKICBjb25zdCBlID0gTWF0aC5wb3coMiwgTWF0aC5jZWlsKE1hdGgubG9nMihDLnZlcnRleENvdW50KSkpOwogIGUgPiB0ICYmICh0ID4gMCAmJiAoZy5fZnJlZShZKSwgZy5fZnJlZShHKSwgZy5fZnJlZShSKSwgZy5fZnJlZShVKSwgZy5fZnJlZShLKSwgZy5fZnJlZShTKSwgZy5fZnJlZSh1KSwgZy5fZnJlZShNKSwgZy5fZnJlZShkKSksIHQgPSBlLCBZID0gZy5fbWFsbG9jKDMgKiB0ICogNCksIEcgPSBnLl9tYWxsb2MoNCAqIHQgKiA0KSwgUiA9IGcuX21hbGxvYygzICogdCAqIDQpLCBVID0gZy5fbWFsbG9jKDQgKiB0KSwgSyA9IGcuX21hbGxvYyh0KSwgUyA9IGcuX21hbGxvYyg4ICogdCAqIDQpLCB1ID0gZy5fbWFsbG9jKDMgKiB0ICogNCksIE0gPSBnLl9tYWxsb2MoNCAqIHQgKiA0KSwgZCA9IGcuX21hbGxvYygzICogdCAqIDQpKSwgZy5IRUFQRjMyLnNldChDLnBvc2l0aW9ucywgWSAvIDQpLCBnLkhFQVBGMzIuc2V0KEMucm90YXRpb25zLCBHIC8gNCksIGcuSEVBUEYzMi5zZXQoQy5zY2FsZXMsIFIgLyA0KSwgZy5IRUFQVTguc2V0KEMuY29sb3JzLCBVKSwgZy5IRUFQVTguc2V0KEMuc2VsZWN0aW9uLCBLKSwgZy5fcGFjaygKICAgIEMuc2VsZWN0ZWQsCiAgICBDLnZlcnRleENvdW50LAogICAgWSwKICAgIEcsCiAgICBSLAogICAgVSwKICAgIEssCiAgICBTLAogICAgdSwKICAgIE0sCiAgICBkCiAgKTsKICBjb25zdCBBID0gbmV3IFVpbnQzMkFycmF5KGcuSEVBUFUzMi5idWZmZXIsIFMsIEMudmVydGV4Q291bnQgKiA4KSwgYyA9IG5ldyBVaW50MzJBcnJheShBLnNsaWNlKCkuYnVmZmVyKSwgSCA9IG5ldyBGbG9hdDMyQXJyYXkoZy5IRUFQRjMyLmJ1ZmZlciwgdSwgQy52ZXJ0ZXhDb3VudCAqIDMpLCB3ID0gbmV3IEZsb2F0MzJBcnJheShILnNsaWNlKCkuYnVmZmVyKSwgaCA9IG5ldyBGbG9hdDMyQXJyYXkoZy5IRUFQRjMyLmJ1ZmZlciwgTSwgQy52ZXJ0ZXhDb3VudCAqIDQpLCBsID0gbmV3IEZsb2F0MzJBcnJheShoLnNsaWNlKCkuYnVmZmVyKSwgeSA9IG5ldyBGbG9hdDMyQXJyYXkoZy5IRUFQRjMyLmJ1ZmZlciwgZCwgQy52ZXJ0ZXhDb3VudCAqIDMpLCBOID0gbmV3IEZsb2F0MzJBcnJheSh5LnNsaWNlKCkuYnVmZmVyKSwgaSA9IHsKICAgIGRhdGE6IGMsCiAgICB3b3JsZFBvc2l0aW9uczogdywKICAgIHdvcmxkUm90YXRpb25zOiBsLAogICAgd29ybGRTY2FsZXM6IE4sCiAgICBvZmZzZXQ6IEMub2Zmc2V0LAogICAgdmVydGV4Q291bnQ6IEMudmVydGV4Q291bnQsCiAgICBwb3NpdGlvbnM6IEMucG9zaXRpb25zLmJ1ZmZlciwKICAgIHJvdGF0aW9uczogQy5yb3RhdGlvbnMuYnVmZmVyLAogICAgc2NhbGVzOiBDLnNjYWxlcy5idWZmZXIsCiAgICBjb2xvcnM6IEMuY29sb3JzLmJ1ZmZlciwKICAgIHNlbGVjdGlvbjogQy5zZWxlY3Rpb24uYnVmZmVyCiAgfTsKICBzZWxmLnBvc3RNZXNzYWdlKHsgcmVzcG9uc2U6IGkgfSwgWwogICAgaS5kYXRhLmJ1ZmZlciwKICAgIGkud29ybGRQb3NpdGlvbnMuYnVmZmVyLAogICAgaS53b3JsZFJvdGF0aW9ucy5idWZmZXIsCiAgICBpLndvcmxkU2NhbGVzLmJ1ZmZlciwKICAgIGkucG9zaXRpb25zLAogICAgaS5yb3RhdGlvbnMsCiAgICBpLnNjYWxlcywKICAgIGkuY29sb3JzLAogICAgaS5zZWxlY3Rpb24KICBdKSwgTCA9ICExOwp9LCBYID0gKCkgPT4gewogIGlmIChGLmxlbmd0aCAhPT0gMCAmJiAhTCkgewogICAgTCA9ICEwOwogICAgY29uc3QgQyA9IEYuc2hpZnQoKTsKICAgIGlBKEMpLCBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgTCA9ICExLCBYKCk7CiAgICB9LCAwKTsKICB9Cn07CnNlbGYub25tZXNzYWdlID0gKEMpID0+IHsKICBpZiAoQy5kYXRhLnNwbGF0KSB7CiAgICBjb25zdCBlID0gQy5kYXRhLnNwbGF0OwogICAgZm9yIChjb25zdCBbQSwgY10gb2YgRi5lbnRyaWVzKCkpCiAgICAgIGlmIChjLm9mZnNldCA9PT0gZS5vZmZzZXQpIHsKICAgICAgICBGW0FdID0gZTsKICAgICAgICByZXR1cm47CiAgICAgIH0KICAgIEYucHVzaChlKSwgWCgpOwogIH0KfTsKLy8jIHNvdXJjZU1hcHBpbmdVUkw9RGF0YVdvcmtlci1ybEdzYXhzNi5qcy5tYXAK", St = (E) => Uint8Array.from(atob(E), (t) => t.charCodeAt(0)), rt = typeof self < "u" && self.Blob && new Blob(["URL.revokeObjectURL(import.meta.url);", St(Ft)], { type: "text/javascript;charset=utf-8" });
function ut(E) {
  let t;
  try {
    if (t = rt && (self.URL || self.webkitURL).createObjectURL(rt), !t) throw "";
    const n = new Worker(t, {
      type: "module",
      name: E?.name
    });
    return n.addEventListener("error", () => {
      (self.URL || self.webkitURL).revokeObjectURL(t);
    }), n;
  } catch {
    return new Worker(
      "data:text/javascript;base64," + Ft,
      {
        type: "module",
        name: E?.name
      }
    );
  }
}
async function Jt(E = {}) {
  var t, n = E, i = import.meta.url, e = "", Q;
  {
    try {
      e = new URL(".", i).href;
    } catch {
    }
    Q = (R) => {
      var N = new XMLHttpRequest();
      return N.open("GET", R, !1), N.responseType = "arraybuffer", N.send(null), new Uint8Array(N.response);
    };
  }
  console.log.bind(console), console.error.bind(console);
  var o, s, r, A, I = !1;
  function d() {
    var R = r.buffer;
    n.HEAPU8 = A = new Uint8Array(R), n.HEAPU32 = new Uint32Array(R), n.HEAPF32 = new Float32Array(R), new BigInt64Array(R), new BigUint64Array(R);
  }
  function U() {
    if (n.preRun)
      for (typeof n.preRun == "function" && (n.preRun = [n.preRun]); n.preRun.length; )
        J(n.preRun.shift());
    S(M);
  }
  function a() {
    I = !0, w.c();
  }
  function B() {
    if (n.postRun)
      for (typeof n.postRun == "function" && (n.postRun = [n.postRun]); n.postRun.length; )
        Z(n.postRun.shift());
    S(y);
  }
  var l;
  function F() {
    return P("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");
  }
  function C(R) {
    if (ArrayBuffer.isView(R))
      return R;
    if (R == l && o)
      return new Uint8Array(o);
    if (Q)
      return Q(R);
    throw 'sync fetching of the wasm failed: you can preload it to Module["wasmBinary"] manually, or emcc.py will do that for you when generating HTML (but not JS)';
  }
  function h(R, N) {
    var f, p = C(R);
    f = new WebAssembly.Module(p);
    var T = new WebAssembly.Instance(f, N);
    return [T, f];
  }
  function V() {
    return { a: W };
  }
  function u() {
    function R(p, T) {
      return w = p.exports, r = w.b, d(), g(w), w;
    }
    var N = V();
    if (n.instantiateWasm)
      return new Promise((p, T) => {
        n.instantiateWasm(N, (X, _) => {
          p(R(X));
        });
      });
    l ??= F();
    var f = h(l, N);
    return R(f[0]);
  }
  for (var S = (R) => {
    for (; R.length > 0; )
      R.shift()(n);
  }, y = [], Z = (R) => y.push(R), M = [], J = (R) => M.push(R), P = (R) => {
    for (var N, f, p = 0, T = 0, X = R.length, _ = new Uint8Array((X * 3 >> 2) - (R[X - 2] == "=") - (R[X - 1] == "=")); p < X; p += 4, T += 3)
      N = G[R.charCodeAt(p + 1)], f = G[R.charCodeAt(p + 2)], _[T] = G[R.charCodeAt(p)] << 2 | N >> 4, _[T + 1] = N << 4 | f >> 2, _[T + 2] = f << 6 | G[R.charCodeAt(p + 3)];
    return _;
  }, v = () => 2147483648, L = (R, N) => Math.ceil(R / N) * N, m = (R) => {
    var N = r.buffer.byteLength, f = (R - N + 65535) / 65536 | 0;
    try {
      return r.grow(f), d(), 1;
    } catch {
    }
  }, x = (R) => {
    var N = A.length;
    R >>>= 0;
    var f = v();
    if (R > f)
      return !1;
    for (var p = 1; p <= 4; p *= 2) {
      var T = N * (1 + 0.2 / p);
      T = Math.min(T, R + 100663296);
      var X = Math.min(f, L(Math.max(R, T), 65536)), _ = m(X);
      if (_)
        return !0;
    }
    return !1;
  }, G = new Uint8Array(123), D = 25; D >= 0; --D)
    G[48 + D] = 52 + D, G[65 + D] = D, G[97 + D] = 26 + D;
  if (G[43] = 62, G[47] = 63, n.noExitRuntime && n.noExitRuntime, n.print && n.print, n.printErr && n.printErr, n.wasmBinary && (o = n.wasmBinary), n.arguments && n.arguments, n.thisProgram && n.thisProgram, n.preInit)
    for (typeof n.preInit == "function" && (n.preInit = [n.preInit]); n.preInit.length > 0; )
      n.preInit.shift()();
  function g(R) {
    n._pack = R.d, n._malloc = R.e, n._free = R.f;
  }
  var W = { a: x };
  function k() {
    U();
    function R() {
      n.calledRun = !0, a(), s?.(n), n.onRuntimeInitialized?.(), B();
    }
    n.setStatus ? (n.setStatus("Running..."), setTimeout(() => {
      setTimeout(() => n.setStatus(""), 1), R();
    }, 1)) : R();
  }
  var w;
  return w = u(), k(), I ? t = n : t = new Promise((R, N) => {
    s = R;
  }), t;
}
const mt = () => new ut();
class Qt {
  constructor(t) {
    this.dataChanged = !1, this.transformsChanged = !1, this.colorTransformsChanged = !1, this._updating = /* @__PURE__ */ new Set(), this._dirty = /* @__PURE__ */ new Set();
    let n = 0, i = 0;
    this._splatIndices = /* @__PURE__ */ new Map(), this._offsets = /* @__PURE__ */ new Map();
    const e = /* @__PURE__ */ new Map();
    for (const U of t.objects)
      U instanceof j && (this._splatIndices.set(U, i), this._offsets.set(U, n), e.set(n, U), n += U.data.vertexCount, i++);
    this._vertexCount = n, this._width = 2048, this._height = Math.ceil(2 * this.vertexCount / this.width), this._data = new Uint32Array(this.width * this.height * 4), this._transformsWidth = 5, this._transformsHeight = e.size, this._transforms = new Float32Array(this._transformsWidth * this._transformsHeight * 4), this._transformIndicesWidth = 1024, this._transformIndicesHeight = Math.ceil(this.vertexCount / this._transformIndicesWidth), this._transformIndices = new Uint32Array(this._transformIndicesWidth * this._transformIndicesHeight), this._colorTransformsWidth = 4, this._colorTransformsHeight = 64, this._colorTransforms = new Float32Array(this._colorTransformsWidth * this._colorTransformsHeight * 4), this._colorTransforms.fill(0), this._colorTransforms[0] = 1, this._colorTransforms[5] = 1, this._colorTransforms[10] = 1, this._colorTransforms[15] = 1, this._colorTransformIndicesWidth = 1024, this._colorTransformIndicesHeight = Math.ceil(this.vertexCount / this._colorTransformIndicesWidth), this._colorTransformIndices = new Uint32Array(
      this._colorTransformIndicesWidth * this._colorTransformIndicesHeight
    ), this.colorTransformIndices.fill(0), this._positions = new Float32Array(this.vertexCount * 3), this._rotations = new Float32Array(this.vertexCount * 4), this._scales = new Float32Array(this.vertexCount * 3), this._worker = mt();
    const Q = (U) => {
      const a = this._splatIndices.get(U);
      this._transforms.set(U.transform.buffer, a * 20), this._transforms[a * 20 + 16] = U.selected ? 1 : 0, U.positionChanged = !1, U.rotationChanged = !1, U.scaleChanged = !1, U.selectedChanged = !1, this.transformsChanged = !0;
    }, o = () => {
      let U = !1;
      for (const l of this._splatIndices.keys())
        if (l.colorTransformChanged) {
          U = !0;
          break;
        }
      if (!U)
        return;
      const a = [new z()];
      this._colorTransformIndices.fill(0);
      let B = 1;
      for (const l of this._splatIndices.keys()) {
        const F = this._offsets.get(l);
        for (const C of l.colorTransforms)
          a.includes(C) || (a.push(C), B++);
        for (const C of l.colorTransformsMap.keys()) {
          const h = l.colorTransformsMap.get(C);
          this._colorTransformIndices[C + F] = h + B - 1;
        }
        l.colorTransformChanged = !1;
      }
      for (let l = 0; l < a.length; l++) {
        const F = a[l];
        this._colorTransforms.set(F.buffer, l * 16);
      }
      this.colorTransformsChanged = !0;
    };
    this._worker.onmessage = (U) => {
      if (U.data.response) {
        const a = U.data.response, B = e.get(a.offset);
        Q(B), o();
        const l = this._splatIndices.get(B);
        for (let F = 0; F < B.data.vertexCount; F++)
          this._transformIndices[a.offset + F] = l;
        this._data.set(a.data, a.offset * 8), B.data.reattach(
          a.positions,
          a.rotations,
          a.scales,
          a.colors,
          a.selection
        ), this._positions.set(a.worldPositions, a.offset * 3), this._rotations.set(a.worldRotations, a.offset * 4), this._scales.set(a.worldScales, a.offset * 3), this._updating.delete(B), B.selectedChanged = !1, this.dataChanged = !0;
      }
    };
    let s;
    async function r() {
      s = await Jt();
    }
    r();
    async function A() {
      for (; !s; )
        await new Promise((U) => setTimeout(U, 0));
    }
    const I = (U) => {
      if (!s) {
        A().then(() => {
          I(U);
        });
        return;
      }
      Q(U);
      const a = s._malloc(3 * U.data.vertexCount * 4), B = s._malloc(4 * U.data.vertexCount * 4), l = s._malloc(3 * U.data.vertexCount * 4), F = s._malloc(4 * U.data.vertexCount), C = s._malloc(U.data.vertexCount), h = s._malloc(8 * U.data.vertexCount * 4), V = s._malloc(3 * U.data.vertexCount * 4), u = s._malloc(4 * U.data.vertexCount * 4), S = s._malloc(3 * U.data.vertexCount * 4);
      s.HEAPF32.set(U.data.positions, a / 4), s.HEAPF32.set(U.data.rotations, B / 4), s.HEAPF32.set(U.data.scales, l / 4), s.HEAPU8.set(U.data.colors, F), s.HEAPU8.set(U.data.selection, C), s._pack(
        U.selected,
        U.data.vertexCount,
        a,
        B,
        l,
        F,
        C,
        h,
        V,
        u,
        S
      );
      const y = new Uint32Array(s.HEAPU32.buffer, h, U.data.vertexCount * 8), Z = new Float32Array(
        s.HEAPF32.buffer,
        V,
        U.data.vertexCount * 3
      ), M = new Float32Array(
        s.HEAPF32.buffer,
        u,
        U.data.vertexCount * 4
      ), J = new Float32Array(s.HEAPF32.buffer, S, U.data.vertexCount * 3), P = this._splatIndices.get(U), v = this._offsets.get(U);
      for (let L = 0; L < U.data.vertexCount; L++)
        this._transformIndices[v + L] = P;
      this._data.set(y, v * 8), this._positions.set(Z, v * 3), this._rotations.set(M, v * 4), this._scales.set(J, v * 3), s._free(a), s._free(B), s._free(l), s._free(F), s._free(C), s._free(h), s._free(V), s._free(u), s._free(S), this.dataChanged = !0, this.colorTransformsChanged = !0;
    }, d = (U) => {
      if ((U.positionChanged || U.rotationChanged || U.scaleChanged || U.selectedChanged) && Q(U), U.colorTransformChanged && o(), !U.data.changed || U.data.detached) return;
      const a = {
        position: new Float32Array(U.position.flat()),
        rotation: new Float32Array(U.rotation.flat()),
        scale: new Float32Array(U.scale.flat()),
        selected: U.selected,
        vertexCount: U.data.vertexCount,
        positions: U.data.positions,
        rotations: U.data.rotations,
        scales: U.data.scales,
        colors: U.data.colors,
        selection: U.data.selection,
        offset: this._offsets.get(U)
      };
      this._worker.postMessage(
        {
          splat: a
        },
        [
          a.position.buffer,
          a.rotation.buffer,
          a.scale.buffer,
          a.positions.buffer,
          a.rotations.buffer,
          a.scales.buffer,
          a.colors.buffer,
          a.selection.buffer
        ]
      ), this._updating.add(U), U.data.detached = !0;
    };
    this.getSplat = (U) => {
      let a = null;
      for (const [B, l] of this._offsets)
        if (U >= l)
          a = B;
        else
          break;
      return a;
    }, this.getLocalIndex = (U, a) => {
      const B = this._offsets.get(U);
      return a - B;
    }, this.markDirty = (U) => {
      this._dirty.add(U);
    }, this.rebuild = () => {
      for (const U of this._dirty)
        d(U);
      this._dirty.clear();
    }, this.dispose = () => {
      this._worker.terminate();
    };
    for (const U of this._splatIndices.keys())
      I(U);
    o();
  }
  get offsets() {
    return this._offsets;
  }
  get data() {
    return this._data;
  }
  get width() {
    return this._width;
  }
  get height() {
    return this._height;
  }
  get transforms() {
    return this._transforms;
  }
  get transformsWidth() {
    return this._transformsWidth;
  }
  get transformsHeight() {
    return this._transformsHeight;
  }
  get transformIndices() {
    return this._transformIndices;
  }
  get transformIndicesWidth() {
    return this._transformIndicesWidth;
  }
  get transformIndicesHeight() {
    return this._transformIndicesHeight;
  }
  get colorTransforms() {
    return this._colorTransforms;
  }
  get colorTransformsWidth() {
    return this._colorTransformsWidth;
  }
  get colorTransformsHeight() {
    return this._colorTransformsHeight;
  }
  get colorTransformIndices() {
    return this._colorTransformIndices;
  }
  get colorTransformIndicesWidth() {
    return this._colorTransformIndicesWidth;
  }
  get colorTransformIndicesHeight() {
    return this._colorTransformIndicesHeight;
  }
  get positions() {
    return this._positions;
  }
  get rotations() {
    return this._rotations;
  }
  get scales() {
    return this._scales;
  }
  get vertexCount() {
    return this._vertexCount;
  }
  get needsRebuild() {
    return this._dirty.size > 0;
  }
  get updating() {
    return this._updating.size > 0;
  }
}
class Bt {
  constructor(t = 0, n = 0, i = 0, e = 255) {
    this.r = t, this.g = n, this.b = i, this.a = e;
  }
  flat() {
    return [this.r, this.g, this.b, this.a];
  }
  flatNorm() {
    return [this.r / 255, this.g / 255, this.b / 255, this.a / 255];
  }
  toHexString() {
    return "#" + this.flat().map((t) => t.toString(16).padStart(2, "0")).join("");
  }
  toString() {
    return `[${this.flat().join(", ")}]`;
  }
}
const ft = () => new Et(), Vt = (
  /* glsl */
  `#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform highp sampler2D u_transforms;
uniform highp usampler2D u_transformIndices;
uniform highp sampler2D u_colorTransforms;
uniform highp usampler2D u_colorTransformIndices;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

uniform bool useDepthFade;
uniform float depthFade;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;
out float vSize;
out float vSelected;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    float selected = float((cen.w >> 24) & 0xffu);

    uint transformIndex = texelFetch(u_transformIndices, ivec2(uint(index) & 0x3ffu, uint(index) >> 10), 0).x;
    mat4 transform = mat4(
        texelFetch(u_transforms, ivec2(0, transformIndex), 0),
        texelFetch(u_transforms, ivec2(1, transformIndex), 0),
        texelFetch(u_transforms, ivec2(2, transformIndex), 0),
        texelFetch(u_transforms, ivec2(3, transformIndex), 0)
    );

    if (selected < 0.5) {
        selected = texelFetch(u_transforms, ivec2(4, transformIndex), 0).x;
    }

    mat4 viewTransform = view * transform;

    vec4 cam = viewTransform * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -pos2d.w || pos2d.z > pos2d.w || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );

    mat3 T = transpose(mat3(viewTransform)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    //ref: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L110-L111
    cov2d[0][0] += 1.0;
    cov2d[1][1] += 1.0;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if (lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    uint colorTransformIndex = texelFetch(u_colorTransformIndices, ivec2(uint(index) & 0x3ffu, uint(index) >> 10), 0).x;
    mat4 colorTransform = mat4(
        texelFetch(u_colorTransforms, ivec2(0, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(1, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(2, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(3, colorTransformIndex), 0)
    );

    vec4 color = vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vColor = colorTransform * color;

    vPosition = position;
    vSize = length(majorAxis);
    vSelected = selected;

    float scalingFactor = 1.0;

    if (useDepthFade) {
        float depthNorm = (pos2d.z / pos2d.w + 1.0) / 2.0;
        float near = 0.1; float far = 100.0;
        float normalizedDepth = (2.0 * near) / (far + near - depthNorm * (far - near));
        float start = max(normalizedDepth - 0.1, 0.0);
        float end = min(normalizedDepth + 0.1, 1.0);
        scalingFactor = clamp((depthFade - start) / (end - start), 0.0, 1.0);
    }

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis * scalingFactor / viewport
        + position.y * minorAxis * scalingFactor / viewport, 0.0, 1.0);
}
`
), Zt = (
  /* glsl */
  `#version 300 es
precision highp float;

uniform float outlineThickness;
uniform vec4 outlineColor;

in vec4 vColor;
in vec2 vPosition;
in float vSize;
in float vSelected;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);

    if (A < -4.0) discard;

    if (vSelected < 0.5) {
        float B = exp(A) * vColor.a;
        fragColor = vec4(B * vColor.rgb, B);
        return;
    }

    float outlineThreshold = -4.0 + (outlineThickness / vSize);

    if (A < outlineThreshold) {
        fragColor = outlineColor;
    } 
    else {
        float B = exp(A) * vColor.a;
        fragColor = vec4(B * vColor.rgb, B);
    }
}
`
);
class lt extends at {
  constructor(t, n) {
    super(t, n), this._outlineThickness = 10, this._outlineColor = new Bt(255, 165, 0, 255), this._renderData = null, this._depthIndex = new Uint32Array(), this._splatTexture = null, this._worker = null;
    const i = t.canvas, e = t.gl;
    let Q, o, s, r, A, I, d, U, a, B, l, F, C, h, V, u, S, y, Z;
    this._resize = () => {
      this._camera && (this._camera.data.setSize(i.width, i.height), this._camera.update(), Q = e.getUniformLocation(this.program, "projection"), e.uniformMatrix4fv(Q, !1, this._camera.data.projectionMatrix.buffer), o = e.getUniformLocation(this.program, "viewport"), e.uniform2fv(o, new Float32Array([i.width, i.height])));
    };
    const M = () => {
      this._worker = ft(), this._worker.onmessage = (m) => {
        if (m.data.depthIndex) {
          const { depthIndex: x } = m.data;
          this._depthIndex = x, e.bindBuffer(e.ARRAY_BUFFER, Z), e.bufferData(e.ARRAY_BUFFER, x, e.STATIC_DRAW);
        }
      };
    };
    this._initialize = () => {
      if (!this._scene || !this._camera) {
        console.error("Cannot render without scene and camera");
        return;
      }
      this._resize(), this._scene.addEventListener("objectAdded", J), this._scene.addEventListener("objectRemoved", P);
      for (const m of this._scene.objects)
        m instanceof j && m.addEventListener("objectChanged", v);
      this._renderData = new Qt(this._scene), s = e.getUniformLocation(this.program, "focal"), e.uniform2fv(s, new Float32Array([this._camera.data.fx, this._camera.data.fy])), r = e.getUniformLocation(this.program, "view"), e.uniformMatrix4fv(r, !1, this._camera.data.viewMatrix.buffer), B = e.getUniformLocation(this.program, "outlineThickness"), e.uniform1f(B, this.outlineThickness), l = e.getUniformLocation(this.program, "outlineColor"), e.uniform4fv(l, new Float32Array(this.outlineColor.flatNorm())), this._splatTexture = e.createTexture(), A = e.getUniformLocation(this.program, "u_texture"), e.uniform1i(A, 0), h = e.createTexture(), I = e.getUniformLocation(this.program, "u_transforms"), e.uniform1i(I, 1), V = e.createTexture(), d = e.getUniformLocation(this.program, "u_transformIndices"), e.uniform1i(d, 2), u = e.createTexture(), U = e.getUniformLocation(this.program, "u_colorTransforms"), e.uniform1i(U, 3), S = e.createTexture(), a = e.getUniformLocation(
        this.program,
        "u_colorTransformIndices"
      ), e.uniform1i(a, 4), y = e.createBuffer(), e.bindBuffer(e.ARRAY_BUFFER, y), e.bufferData(e.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), e.STATIC_DRAW), F = e.getAttribLocation(this.program, "position"), e.enableVertexAttribArray(F), e.vertexAttribPointer(F, 2, e.FLOAT, !1, 0, 0), Z = e.createBuffer(), C = e.getAttribLocation(this.program, "index"), e.enableVertexAttribArray(C), e.bindBuffer(e.ARRAY_BUFFER, Z), M();
    };
    const J = (m) => {
      const x = m;
      x.object instanceof j && x.object.addEventListener("objectChanged", v), L();
    }, P = (m) => {
      const x = m;
      x.object instanceof j && x.object.removeEventListener("objectChanged", v), L();
    }, v = (m) => {
      const x = m;
      x.object instanceof j && this._renderData && this._renderData.markDirty(x.object);
    }, L = () => {
      this._renderData?.dispose(), this._renderData = new Qt(this._scene), this._worker?.terminate(), M();
    };
    this._render = () => {
      if (!this._scene || !this._camera || !this.renderData) {
        console.error("Cannot render without scene and camera");
        return;
      }
      if (this.renderData.needsRebuild && this.renderData.rebuild(), this.renderData.dataChanged || this.renderData.transformsChanged || this.renderData.colorTransformsChanged) {
        this.renderData.dataChanged && (e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, this.splatTexture), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.RGBA32UI,
          this.renderData.width,
          this.renderData.height,
          0,
          e.RGBA_INTEGER,
          e.UNSIGNED_INT,
          this.renderData.data
        )), this.renderData.transformsChanged && (e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, h), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.RGBA32F,
          this.renderData.transformsWidth,
          this.renderData.transformsHeight,
          0,
          e.RGBA,
          e.FLOAT,
          this.renderData.transforms
        ), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, V), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.R32UI,
          this.renderData.transformIndicesWidth,
          this.renderData.transformIndicesHeight,
          0,
          e.RED_INTEGER,
          e.UNSIGNED_INT,
          this.renderData.transformIndices
        )), this.renderData.colorTransformsChanged && (e.activeTexture(e.TEXTURE3), e.bindTexture(e.TEXTURE_2D, u), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.RGBA32F,
          this.renderData.colorTransformsWidth,
          this.renderData.colorTransformsHeight,
          0,
          e.RGBA,
          e.FLOAT,
          this.renderData.colorTransforms
        ), e.activeTexture(e.TEXTURE4), e.bindTexture(e.TEXTURE_2D, S), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.R32UI,
          this.renderData.colorTransformIndicesWidth,
          this.renderData.colorTransformIndicesHeight,
          0,
          e.RED_INTEGER,
          e.UNSIGNED_INT,
          this.renderData.colorTransformIndices
        ));
        const m = new Float32Array(this.renderData.positions.slice().buffer), x = new Float32Array(this.renderData.transforms.slice().buffer), G = new Uint32Array(this.renderData.transformIndices.slice().buffer);
        this._worker?.postMessage(
          {
            sortData: {
              positions: m,
              transforms: x,
              transformIndices: G,
              vertexCount: this.renderData.vertexCount
            }
          },
          [m.buffer, x.buffer, G.buffer]
        ), this.renderData.dataChanged = !1, this.renderData.transformsChanged = !1, this.renderData.colorTransformsChanged = !1;
      }
      this._camera.update(), this._worker?.postMessage({ viewProj: this._camera.data.viewProj.buffer }), e.viewport(0, 0, i.width, i.height), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.disable(e.DEPTH_TEST), e.enable(e.BLEND), e.blendFuncSeparate(e.ONE_MINUS_DST_ALPHA, e.ONE, e.ONE_MINUS_DST_ALPHA, e.ONE), e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD), e.uniformMatrix4fv(Q, !1, this._camera.data.projectionMatrix.buffer), e.uniformMatrix4fv(r, !1, this._camera.data.viewMatrix.buffer), e.bindBuffer(e.ARRAY_BUFFER, y), e.vertexAttribPointer(F, 2, e.FLOAT, !1, 0, 0), e.bindBuffer(e.ARRAY_BUFFER, Z), e.bufferData(e.ARRAY_BUFFER, this.depthIndex, e.STATIC_DRAW), e.vertexAttribIPointer(C, 1, e.INT, 0, 0), e.vertexAttribDivisor(C, 1), e.drawArraysInstanced(e.TRIANGLE_FAN, 0, 4, this.depthIndex.length);
    }, this._dispose = () => {
      if (!this._scene || !this._camera || !this.renderData) {
        console.error("Cannot dispose without scene and camera");
        return;
      }
      this._scene.removeEventListener("objectAdded", J), this._scene.removeEventListener("objectRemoved", P);
      for (const m of this._scene.objects)
        m instanceof j && m.removeEventListener("objectChanged", v);
      this._worker?.terminate(), this.renderData.dispose(), e.deleteTexture(this.splatTexture), e.deleteTexture(h), e.deleteTexture(V), e.deleteBuffer(Z), e.deleteBuffer(y);
    }, this._setOutlineThickness = (m) => {
      this._outlineThickness = m, this._initialized && e.uniform1f(B, m);
    }, this._setOutlineColor = (m) => {
      this._outlineColor = m, this._initialized && e.uniform4fv(l, new Float32Array(m.flatNorm()));
    };
  }
  get renderData() {
    return this._renderData;
  }
  get depthIndex() {
    return this._depthIndex;
  }
  get splatTexture() {
    return this._splatTexture;
  }
  get outlineThickness() {
    return this._outlineThickness;
  }
  set outlineThickness(t) {
    this._setOutlineThickness(t);
  }
  get outlineColor() {
    return this._outlineColor;
  }
  set outlineColor(t) {
    this._setOutlineColor(t);
  }
  get worker() {
    return this._worker;
  }
  _getVertexSource() {
    return Vt;
  }
  _getFragmentSource() {
    return Zt;
  }
}
class Wt {
  constructor(t = 1) {
    let n = 0, i = !1, e, Q, o, s;
    this.initialize = (r) => {
      if (!(r instanceof lt))
        throw new Error("FadeInPass requires a RenderProgram");
      n = r.started ? 1 : 0, i = !0, e = r, Q = r.renderer.gl, o = Q.getUniformLocation(e.program, "useDepthFade"), Q.uniform1i(o, 1), s = Q.getUniformLocation(e.program, "depthFade"), Q.uniform1f(s, n);
    }, this.render = () => {
      !i || e.renderData?.updating || (Q.useProgram(e.program), n = Math.min(n + t * 0.01, 1), n >= 1 && (i = !1, Q.uniform1i(o, 0)), Q.uniform1f(s, n));
    };
  }
  dispose() {
  }
}
class xt {
  constructor(t = null, n = null) {
    this._backgroundColor = new Bt();
    const i = t || document.createElement("canvas");
    t || (i.style.display = "block", i.style.boxSizing = "border-box", i.style.width = "100%", i.style.height = "100%", i.style.margin = "0", i.style.padding = "0", document.body.appendChild(i)), i.style.background = this._backgroundColor.toHexString(), this._canvas = i, this._gl = i.getContext("webgl2", { antialias: !1 });
    const e = n || [];
    n || e.push(new Wt()), this._renderProgram = new lt(this, e);
    const Q = [this._renderProgram];
    this.resize = () => {
      const o = i.clientWidth, s = i.clientHeight;
      (i.width !== o || i.height !== s) && this.setSize(o, s);
    }, this.setSize = (o, s) => {
      i.width = o, i.height = s, this._gl.viewport(0, 0, i.width, i.height);
      for (const r of Q)
        r.resize();
    }, this.render = (o, s) => {
      for (const r of Q)
        r.render(o, s);
    }, this.dispose = () => {
      for (const o of Q)
        o.dispose();
    }, this.addProgram = (o) => {
      Q.push(o);
    }, this.removeProgram = (o) => {
      const s = Q.indexOf(o);
      if (s < 0)
        throw new Error("Program not found");
      Q.splice(s, 1);
    }, this.resize();
  }
  get canvas() {
    return this._canvas;
  }
  get gl() {
    return this._gl;
  }
  get renderProgram() {
    return this._renderProgram;
  }
  get backgroundColor() {
    return this._backgroundColor;
  }
  set backgroundColor(t) {
    this._backgroundColor = t, this._canvas.style.background = t.toHexString();
  }
}
class Mt {
  constructor(t, n, i = 0.5, e = 0.5, Q = 5, o = !0, s = new c()) {
    this.minAngle = -90, this.maxAngle = 90, this.minZoom = 0.1, this.maxZoom = 30, this.orbitSpeed = 1, this.panSpeed = 1, this.zoomSpeed = 1, this.dampening = 0.12, this.setCameraTarget = () => {
    };
    let r = s.clone(), A = r.clone(), I = i, d = e, U = Q, a = !1, B = !1, l = 0, F = 0, C = 0;
    const h = {};
    let V = !1;
    const u = () => {
      if (V) return;
      const g = t.rotation.toEuler();
      I = -g.y, d = -g.x;
      const W = t.position.x - U * Math.sin(I) * Math.cos(d), k = t.position.y + U * Math.sin(d), w = t.position.z + U * Math.cos(I) * Math.cos(d);
      A = new c(W, k, w);
    };
    t.addEventListener("objectChanged", u), this.setCameraTarget = (g) => {
      const W = g.x - t.position.x, k = g.y - t.position.y, w = g.z - t.position.z;
      U = Math.sqrt(W * W + k * k + w * w), d = Math.atan2(k, Math.sqrt(W * W + w * w)), I = -Math.atan2(W, w), A = new c(g.x, g.y, g.z);
    };
    const S = () => 0.1 + 0.9 * (U - this.minZoom) / (this.maxZoom - this.minZoom), y = (g) => {
      h[g.code] = !0, g.code === "ArrowUp" && (h.KeyW = !0), g.code === "ArrowDown" && (h.KeyS = !0), g.code === "ArrowLeft" && (h.KeyA = !0), g.code === "ArrowRight" && (h.KeyD = !0);
    }, Z = (g) => {
      h[g.code] = !1, g.code === "ArrowUp" && (h.KeyW = !1), g.code === "ArrowDown" && (h.KeyS = !1), g.code === "ArrowLeft" && (h.KeyA = !1), g.code === "ArrowRight" && (h.KeyD = !1);
    }, M = (g) => {
      D(g), a = !0, B = g.button === 2, F = g.clientX, C = g.clientY, window.addEventListener("mouseup", J);
    }, J = (g) => {
      D(g), a = !1, B = !1, window.removeEventListener("mouseup", J);
    }, P = (g) => {
      if (D(g), !a || !t) return;
      const W = g.clientX - F, k = g.clientY - C;
      if (B) {
        const w = S(), R = -W * this.panSpeed * 0.01 * w, N = -k * this.panSpeed * 0.01 * w, f = K.RotationFromQuaternion(t.rotation).buffer, p = new c(f[0], f[3], f[6]), T = new c(f[1], f[4], f[7]);
        A = A.add(p.multiply(R)), A = A.add(T.multiply(N));
      } else
        I -= W * this.orbitSpeed * 3e-3, d += k * this.orbitSpeed * 3e-3, d = Math.min(
          Math.max(d, this.minAngle * Math.PI / 180),
          this.maxAngle * Math.PI / 180
        );
      F = g.clientX, C = g.clientY;
    }, v = (g) => {
      D(g);
      const W = S();
      U += g.deltaY * this.zoomSpeed * 0.025 * W, U = Math.min(Math.max(U, this.minZoom), this.maxZoom);
    }, L = (g) => {
      if (D(g), g.touches.length === 1)
        a = !0, B = !1, F = g.touches[0].clientX, C = g.touches[0].clientY, l = 0;
      else if (g.touches.length === 2) {
        a = !0, B = !0, F = (g.touches[0].clientX + g.touches[1].clientX) / 2, C = (g.touches[0].clientY + g.touches[1].clientY) / 2;
        const W = g.touches[0].clientX - g.touches[1].clientX, k = g.touches[0].clientY - g.touches[1].clientY;
        l = Math.sqrt(W * W + k * k);
      }
    }, m = (g) => {
      D(g), a = !1, B = !1;
    }, x = (g) => {
      if (D(g), !(!a || !t))
        if (B) {
          const W = S(), k = g.touches[0].clientX - g.touches[1].clientX, w = g.touches[0].clientY - g.touches[1].clientY, R = Math.sqrt(k * k + w * w), N = l - R;
          U += N * this.zoomSpeed * 0.1 * W, U = Math.min(Math.max(U, this.minZoom), this.maxZoom), l = R;
          const f = (g.touches[0].clientX + g.touches[1].clientX) / 2, p = (g.touches[0].clientY + g.touches[1].clientY) / 2, T = f - F, X = p - C, _ = K.RotationFromQuaternion(t.rotation).buffer, gt = new c(_[0], _[3], _[6]), dt = new c(_[1], _[4], _[7]);
          A = A.add(gt.multiply(-T * this.panSpeed * 0.025 * W)), A = A.add(dt.multiply(-X * this.panSpeed * 0.025 * W)), F = f, C = p;
        } else {
          const W = g.touches[0].clientX - F, k = g.touches[0].clientY - C;
          I -= W * this.orbitSpeed * 3e-3, d += k * this.orbitSpeed * 3e-3, d = Math.min(
            Math.max(d, this.minAngle * Math.PI / 180),
            this.maxAngle * Math.PI / 180
          ), F = g.touches[0].clientX, C = g.touches[0].clientY;
        }
    }, G = (g, W, k) => (1 - k) * g + k * W;
    this.update = () => {
      V = !0, i = G(i, I, this.dampening), e = G(e, d, this.dampening), Q = G(Q, U, this.dampening), r = r.lerp(A, this.dampening);
      const g = r.x + Q * Math.sin(i) * Math.cos(e), W = r.y - Q * Math.sin(e), k = r.z - Q * Math.cos(i) * Math.cos(e);
      t.position = new c(g, W, k);
      const w = r.subtract(t.position).normalize(), R = Math.asin(-w.y), N = Math.atan2(w.x, w.z);
      t.rotation = b.FromEuler(new c(R, N, 0));
      const f = 0.025, p = 0.01, T = K.RotationFromQuaternion(t.rotation).buffer, X = new c(-T[2], -T[5], -T[8]), _ = new c(T[0], T[3], T[6]);
      h.KeyS && (A = A.add(X.multiply(f))), h.KeyW && (A = A.subtract(X.multiply(f))), h.KeyA && (A = A.subtract(_.multiply(f))), h.KeyD && (A = A.add(_.multiply(f))), h.KeyE && (I += p), h.KeyQ && (I -= p), h.KeyR && (d += p), h.KeyF && (d -= p), V = !1;
    };
    const D = (g) => {
      g.preventDefault(), g.stopPropagation();
    };
    this.dispose = () => {
      n.removeEventListener("dragenter", D), n.removeEventListener("dragover", D), n.removeEventListener("dragleave", D), n.removeEventListener("contextmenu", D), n.removeEventListener("mousedown", M), n.removeEventListener("mousemove", P), n.removeEventListener("wheel", v), n.removeEventListener("touchstart", L), n.removeEventListener("touchend", m), n.removeEventListener("touchmove", x), o && (window.removeEventListener("keydown", y), window.removeEventListener("keyup", Z));
    }, o && (window.addEventListener("keydown", y), window.addEventListener("keyup", Z)), n.addEventListener("dragenter", D), n.addEventListener("dragover", D), n.addEventListener("dragleave", D), n.addEventListener("contextmenu", D), n.addEventListener("mousedown", M), n.addEventListener("mousemove", P), n.addEventListener("wheel", v), n.addEventListener("touchstart", L), n.addEventListener("touchend", m), n.addEventListener("touchmove", x), this.update();
  }
}
class Lt {
  constructor(t, n) {
    this.moveSpeed = 1.5, this.lookSpeed = 0.7, this.dampening = 0.5;
    const i = {};
    let e = t.rotation.toEuler().x, Q = t.rotation.toEuler().y, o = t.position, s = !1;
    const r = () => {
      n.requestPointerLock();
    }, A = () => {
      s = document.pointerLockElement === n, s ? n.addEventListener("mousemove", I) : n.removeEventListener("mousemove", I);
    }, I = (B) => {
      const l = B.movementX, F = B.movementY;
      Q += l * this.lookSpeed * 1e-3, e -= F * this.lookSpeed * 1e-3, e = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, e));
    }, d = (B) => {
      i[B.code] = !0, B.code === "ArrowUp" && (i.KeyW = !0), B.code === "ArrowDown" && (i.KeyS = !0), B.code === "ArrowLeft" && (i.KeyA = !0), B.code === "ArrowRight" && (i.KeyD = !0);
    }, U = (B) => {
      i[B.code] = !1, B.code === "ArrowUp" && (i.KeyW = !1), B.code === "ArrowDown" && (i.KeyS = !1), B.code === "ArrowLeft" && (i.KeyA = !1), B.code === "ArrowRight" && (i.KeyD = !1), B.code === "Escape" && document.exitPointerLock();
    };
    this.update = () => {
      const B = K.RotationFromQuaternion(t.rotation).buffer, l = new c(-B[2], -B[5], -B[8]), F = new c(B[0], B[3], B[6]);
      let C = new c(0, 0, 0);
      i.KeyS && (C = C.add(l)), i.KeyW && (C = C.subtract(l)), i.KeyA && (C = C.subtract(F)), i.KeyD && (C = C.add(F)), C = new c(C.x, 0, C.z), C.magnitude() > 0 && (C = C.normalize()), o = o.add(C.multiply(this.moveSpeed * 0.01)), t.position = t.position.add(o.subtract(t.position).multiply(this.dampening)), t.rotation = b.FromEuler(new c(e, Q, 0));
    };
    const a = (B) => {
      B.preventDefault(), B.stopPropagation();
    };
    this.dispose = () => {
      n.removeEventListener("dragenter", a), n.removeEventListener("dragover", a), n.removeEventListener("dragleave", a), n.removeEventListener("contextmenu", a), n.removeEventListener("mousedown", r), document.removeEventListener("pointerlockchange", A), window.removeEventListener("keydown", d), window.removeEventListener("keyup", U);
    }, window.addEventListener("keydown", d), window.addEventListener("keyup", U), n.addEventListener("dragenter", a), n.addEventListener("dragover", a), n.addEventListener("dragleave", a), n.addEventListener("contextmenu", a), n.addEventListener("mousedown", r), document.addEventListener("pointerlockchange", A), this.update();
  }
}
class Gt {
  constructor(t, n) {
    this.normal = t, this.point = n;
  }
  intersect(t, n) {
    const i = this.normal.dot(n);
    if (Math.abs(i) < 1e-4)
      return null;
    const e = this.normal.dot(this.point.subtract(t)) / i;
    return e < 0 ? null : t.add(n.multiply(e));
  }
}
class vt {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  initialize(t) {
  }
  render() {
  }
  dispose() {
  }
}
const Nt = (
  /* glsl */
  `#version 300 es
precision highp float;
precision highp int;
  
uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
uniform float time;
  
in vec2 position;
in int index;
  
out vec4 vColor;
out vec2 vPosition;
  
void main () {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    uvec4 motion1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 3u, uint(index) >> 10), 0);
    vec2 trbf = unpackHalf2x16(motion1.w);
    float dt = time - trbf.x;

    float topacity = exp(-1.0 * pow(dt / trbf.y, 2.0));
    if(topacity < 0.02) return;

    uvec4 motion0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 2u, uint(index) >> 10), 0);
    uvec4 static0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2), uint(index) >> 10), 0);

    vec2 m0 = unpackHalf2x16(motion0.x), m1 = unpackHalf2x16(motion0.y), m2 = unpackHalf2x16(motion0.z), 
         m3 = unpackHalf2x16(motion0.w), m4 = unpackHalf2x16(motion1.x); 
      
    vec4 trot = vec4(unpackHalf2x16(motion1.y).xy, unpackHalf2x16(motion1.z).xy) * dt;
    vec3 tpos = (vec3(m0.xy, m1.x) * dt + vec3(m1.y, m2.xy) * dt*dt + vec3(m3.xy, m4.x) * dt*dt*dt);
      
    vec4 cam = view * vec4(uintBitsToFloat(static0.xyz) + tpos, 1);
    vec4 pos = projection * cam;
  
    float clip = 1.2 * pos.w;
    if (pos.z < -clip || pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
    uvec4 static1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 1u, uint(index) >> 10), 0);

    vec4 rot = vec4(unpackHalf2x16(static0.w).xy, unpackHalf2x16(static1.x).xy) + trot;
    vec3 scale = vec3(unpackHalf2x16(static1.y).xy, unpackHalf2x16(static1.z).x);
    rot /= sqrt(dot(rot, rot));
  
    mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 R = mat3(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w), 2.0 * (rot.y * rot.w + rot.x * rot.z),
        2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w - rot.x * rot.y),
        2.0 * (rot.y * rot.w - rot.x * rot.z), 2.0 * (rot.z * rot.w + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z));
    mat3 M = S * R;
    mat3 Vrk = 4.0 * transpose(M) * M;
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );
  
    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;
  
    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;
  
    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
      
    uint rgba = static1.w;
    vColor = 
        clamp(pos.z/pos.w+1.0, 0.0, 1.0) * 
        vec4(1.0, 1.0, 1.0, topacity) *
        vec4(
            (rgba) & 0xffu, 
            (rgba >> 8) & 0xffu, 
            (rgba >> 16) & 0xffu, 
            (rgba >> 24) & 0xffu) / 255.0;

    vec2 vCenter = vec2(pos) / pos.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis / viewport 
        + position.y * minorAxis / viewport, 0.0, 1.0);

    vPosition = position;
}
`
), pt = (
  /* glsl */
  `#version 300 es
precision highp float;
  
in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}
`
);
class _t extends at {
  constructor(t, n = []) {
    super(t, n), this._renderData = null, this._depthIndex = new Uint32Array(), this._splatTexture = null;
    const i = t.canvas, e = t.gl;
    let Q, o, s, r, A, I, d, U, a, B, l;
    this._resize = () => {
      this._camera && (this._camera.data.setSize(i.width, i.height), this._camera.update(), o = e.getUniformLocation(this.program, "projection"), e.uniformMatrix4fv(o, !1, this._camera.data.projectionMatrix.buffer), s = e.getUniformLocation(this.program, "viewport"), e.uniform2fv(s, new Float32Array([i.width, i.height])));
    };
    const F = () => {
      if (t.renderProgram.worker === null) {
        console.error("Render program is not initialized. Cannot render without worker");
        return;
      }
      Q = t.renderProgram.worker, Q.onmessage = (u) => {
        if (u.data.depthIndex) {
          const { depthIndex: S } = u.data;
          this._depthIndex = S, e.bindBuffer(e.ARRAY_BUFFER, l), e.bufferData(e.ARRAY_BUFFER, S, e.STATIC_DRAW);
        }
      };
    };
    this._initialize = () => {
      if (!this._scene || !this._camera) {
        console.error("Cannot render without scene and camera");
        return;
      }
      this._resize(), this._scene.addEventListener("objectAdded", C), this._scene.addEventListener("objectRemoved", h);
      for (const Z of this._scene.objects)
        Z instanceof O && (this._renderData === null ? (this._renderData = Z.data, Z.addEventListener("objectChanged", V)) : console.warn("Multiple Splatv objects are not currently supported"));
      if (this._renderData === null) {
        console.error("Cannot render without Splatv object");
        return;
      }
      r = e.getUniformLocation(this.program, "focal"), e.uniform2fv(r, new Float32Array([this._camera.data.fx, this._camera.data.fy])), A = e.getUniformLocation(this.program, "view"), e.uniformMatrix4fv(A, !1, this._camera.data.viewMatrix.buffer), this._splatTexture = e.createTexture(), I = e.getUniformLocation(this.program, "u_texture"), e.uniform1i(I, 0), d = e.getUniformLocation(this.program, "time"), e.uniform1f(d, Math.sin(Date.now() / 1e3) / 2 + 1 / 2), B = e.createBuffer(), e.bindBuffer(e.ARRAY_BUFFER, B), e.bufferData(e.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), e.STATIC_DRAW), U = e.getAttribLocation(this.program, "position"), e.enableVertexAttribArray(U), e.vertexAttribPointer(U, 2, e.FLOAT, !1, 0, 0), l = e.createBuffer(), a = e.getAttribLocation(this.program, "index"), e.enableVertexAttribArray(a), e.bindBuffer(e.ARRAY_BUFFER, l), F(), e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, this._splatTexture), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
        e.TEXTURE_2D,
        0,
        e.RGBA32UI,
        this._renderData.width,
        this._renderData.height,
        0,
        e.RGBA_INTEGER,
        e.UNSIGNED_INT,
        this._renderData.data
      );
      const u = this._renderData.positions, S = new Float32Array(new z().buffer), y = new Uint32Array(this._renderData.vertexCount);
      y.fill(0), Q.postMessage(
        {
          sortData: {
            positions: u,
            transforms: S,
            transformIndices: y,
            vertexCount: this._renderData.vertexCount
          }
        },
        [u.buffer, S.buffer, y.buffer]
      );
    };
    const C = (u) => {
      const S = u;
      S.object instanceof O && (this._renderData === null ? (this._renderData = S.object.data, S.object.addEventListener("objectChanged", V)) : console.warn("Splatv not supported by default RenderProgram. Use VideoRenderProgram instead.")), this.dispose();
    }, h = (u) => {
      const S = u;
      S.object instanceof O && this._renderData === S.object.data && (this._renderData = null, S.object.removeEventListener("objectChanged", V)), this.dispose();
    }, V = (u) => {
      const S = u;
      S.object instanceof O && this._renderData === S.object.data && this.dispose();
    };
    this._render = () => {
      if (!this._scene || !this._camera) {
        console.error("Cannot render without scene and camera");
        return;
      }
      if (!this._renderData) {
        console.warn("Cannot render without Splatv object");
        return;
      }
      this._camera.update(), Q.postMessage({ viewProj: this._camera.data.viewProj.buffer }), e.viewport(0, 0, i.width, i.height), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.disable(e.DEPTH_TEST), e.enable(e.BLEND), e.blendFuncSeparate(e.ONE_MINUS_DST_ALPHA, e.ONE, e.ONE_MINUS_DST_ALPHA, e.ONE), e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD), e.uniformMatrix4fv(o, !1, this._camera.data.projectionMatrix.buffer), e.uniformMatrix4fv(A, !1, this._camera.data.viewMatrix.buffer), e.uniform1f(d, Math.sin(Date.now() / 1e3) / 2 + 1 / 2), e.bindBuffer(e.ARRAY_BUFFER, B), e.vertexAttribPointer(U, 2, e.FLOAT, !1, 0, 0), e.bindBuffer(e.ARRAY_BUFFER, l), e.bufferData(e.ARRAY_BUFFER, this._depthIndex, e.STATIC_DRAW), e.vertexAttribIPointer(a, 1, e.INT, 0, 0), e.vertexAttribDivisor(a, 1), e.drawArraysInstanced(e.TRIANGLE_FAN, 0, 4, this._renderData.vertexCount);
    }, this._dispose = () => {
      if (!this._scene || !this._camera) {
        console.error("Cannot dispose without scene and camera");
        return;
      }
      this._scene.removeEventListener("objectAdded", C), this._scene.removeEventListener("objectRemoved", h);
      for (const u of this._scene.objects)
        u instanceof O && this._renderData === u.data && (this._renderData = null, u.removeEventListener("objectChanged", V));
      Q?.terminate(), e.deleteTexture(this._splatTexture), e.deleteBuffer(l), e.deleteBuffer(B);
    };
  }
  get renderData() {
    return this._renderData;
  }
  _getVertexSource() {
    return Nt;
  }
  _getFragmentSource() {
    return pt;
  }
}
class tt {
  constructor(t, n, i) {
    this.bounds = t, this.boxes = n, this.left = null, this.right = null, this.pointIndices = [], i.length > 1 ? this.split(t, n, i) : i.length > 0 && (this.pointIndices = i);
  }
  split(t, n, i) {
    const e = t.size().maxComponent();
    i.sort((r, A) => n[r].center().getComponent(e) - n[A].center().getComponent(e));
    const Q = Math.floor(i.length / 2), o = i.slice(0, Q), s = i.slice(Q);
    this.left = new tt(t, n, o), this.right = new tt(t, n, s);
  }
  queryRange(t) {
    return this.bounds.intersects(t) ? this.left !== null && this.right !== null ? this.left.queryRange(t).concat(this.right.queryRange(t)) : this.pointIndices.filter((n) => t.intersects(this.boxes[n])) : [];
  }
}
class Dt {
  constructor(t, n) {
    const i = n.map((e, Q) => Q);
    this.root = new tt(t, n, i);
  }
  queryRange(t) {
    return this.root.queryRange(t);
  }
}
class Kt {
  constructor(t, n = 100, i = 1) {
    let e = 0, Q = null, o = [];
    const s = () => {
      if (t.renderData === null) {
        console.error("IntersectionTester cannot be called before renderProgram has been initialized");
        return;
      }
      o = [];
      const r = t.renderData, A = new Array(r.offsets.size);
      let I = 0;
      const d = new $(
        new c(1 / 0, 1 / 0, 1 / 0),
        new c(-1 / 0, -1 / 0, -1 / 0)
      );
      for (const U of r.offsets.keys()) {
        const a = U.bounds;
        A[I++] = a, d.expand(a.min), d.expand(a.max), o.push(U);
      }
      d.permute(), Q = new Dt(d, A), e = r.vertexCount;
    };
    this.testPoint = (r, A) => {
      if (t.renderData === null || t.camera === null)
        return console.error("IntersectionTester cannot be called before renderProgram has been initialized"), null;
      if (s(), Q === null)
        return console.error("Failed to build octree for IntersectionTester"), null;
      const I = t.renderData, d = t.camera;
      e !== I.vertexCount && console.warn("IntersectionTester has not been rebuilt since the last render");
      const U = d.screenPointToRay(r, A);
      for (let a = 0; a < n; a += i) {
        const B = d.position.add(U.multiply(a)), l = new c(
          B.x - i / 2,
          B.y - i / 2,
          B.z - i / 2
        ), F = new c(
          B.x + i / 2,
          B.y + i / 2,
          B.z + i / 2
        ), C = new $(l, F), h = Q.queryRange(C);
        if (h.length > 0)
          return o[h[0]];
      }
      return null;
    };
  }
}
export {
  yt as Camera,
  ct as CameraData,
  Bt as Color32,
  Lt as FPSControls,
  Wt as FadeInPass,
  Kt as IntersectionTester,
  kt as Loader,
  K as Matrix3,
  z as Matrix4,
  et as Object3D,
  Mt as OrbitControls,
  Tt as PLYLoader,
  Gt as Plane,
  b as Quaternion,
  Qt as RenderData,
  lt as RenderProgram,
  wt as Scene,
  vt as ShaderPass,
  at as ShaderProgram,
  j as Splat,
  Y as SplatData,
  O as Splatv,
  nt as SplatvData,
  bt as SplatvLoader,
  c as Vector3,
  H as Vector4,
  _t as VideoRenderProgram,
  xt as WebGLRenderer
};
//# sourceMappingURL=index.es.js.map
