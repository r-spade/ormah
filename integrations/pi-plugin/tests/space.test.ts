import assert from "node:assert/strict";
import test from "node:test";
import { resolveSpace } from "../src/space.js";

test("explicit null selects global scope even when ORMAH_SPACE is set", () => {
	const previous = process.env.ORMAH_SPACE;
	process.env.ORMAH_SPACE = "project";
	try {
		assert.equal(resolveSpace(null, "/tmp/project"), null);
	} finally {
		if (previous === undefined) delete process.env.ORMAH_SPACE;
		else process.env.ORMAH_SPACE = previous;
	}
});
