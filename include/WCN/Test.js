const Module = require("module");

const mem = Module._wcn_math_Vec2xN_malloc(4);
// . wcn_math_Vec2xN_set_by_vec2 // 通过栈临时值 不导出于wasm
// ._wcn_math_Vec2xN_set_by_vec2_ptr // 通过vec2指针 导出于wasm
// ._wcn_math_Vec2xN_set_by_xy //通过js与wasm公用基本类型临时值 导出于wasm
Module._wcn_math_Vec2xN_set_by_xy(mem, 0, 0, 1);
Module._wcn_math_Vec2xN_set_by_xy(mem, 1, 1, 1);
Module._wcn_math_Vec2xN_set_by_xy(mem, 2, 0, 0);
Module._wcn_math_Vec2xN_set_by_xy(mem, 3, 1, 0);

class Vec2xN {
  #memPtr;
  #size;
  #length = 0;
  get length() {
    return this.#length;
  }
  get maxSize() {
    return this.#size;
  }
  get isFull() {
    return this.length;
  }
  set length(value) {
    if (value === 0) {
      Module._wcn_math_Vec2xN_free(this.#memPtr);
      this.#memPtr = 0;
      this.#length = 0;
    } else {
      throw new Error("You con't set length for the vec2xN");
    }
  }
  static malloc(size) {
    const self = new Vec2xN();
    self.#size = size;
    self.#memPtr = Module._wcn_math_Vec2xN_malloc(size);
    return self;
  }
  pushBack(x, y) {
    if (this.#length >= this.#size) {
      throw new Error("The's Vec2xN is Full");
    }
    this.#length += 1;
    const index = this.#length - 1;
    Module._wcn_math_Vec2xN_set_by_xy(this.#memPtr, index, x, y);
    return index;
  }
  getVec2PtrByIndex(vec2_ptr) {
    Module._wcn_math_Vec2xN_get(vec2_ptr, this.#memPtr, 3);
  }
  free() {
    Module._wcn_math_Vec2xN_free(this.#memPtr);
    this.#memPtr = 0;
    this.#length = 0;
  }
}
function main() {
  const vec2x4 = Vec2xN.malloc(4);
  {
    vec2x4.pushBack(1, 2);
    vec2x4.pushBack(1, 2);
    vec2x4.pushBack(1, 2);
    vec2x4.pushBack(1, 2);
  }
  const vec2_ptr = Module._malloc(8); // f32 * 2;
  vec2x4.getVec2PtrByIndex(vec2_ptr, 0);
}
