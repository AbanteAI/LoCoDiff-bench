//! This file renders underline sprites. To draw underlines, we render the
//! full cell-width as a sprite and then draw it as a separate pass to the
//! text.
//!
//! We used to render the underlines directly in the GPU shaders but its
//! annoying to support multiple types of underlines and its also annoying
//! to maintain and debug another set of shaders for each renderer instead of
//! just relying on the glyph system we already need to support for text
//! anyways.
//!
//! This also renders strikethrough, so its really more generally a
//! "horizontal line" renderer.
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const font = @import("../main.zig");
const Sprite = font.sprite.Sprite;

/// Draw an underline.
pub fn renderGlyph(
    alloc: Allocator,
    atlas: *font.Atlas,
    sprite: Sprite,
    width: u32,
    height: u32,
    line_pos: u32,
    line_thickness: u32,
) !font.Glyph {
    // Draw the appropriate sprite
    var canvas: font.sprite.Canvas, const offset_y: i32 = switch (sprite) {
        .underline => try drawSingle(alloc, width, line_thickness),
        .underline_double => try drawDouble(alloc, width, line_thickness),
        .underline_dotted => try drawDotted(alloc, width, line_thickness),
        .underline_dashed => try drawDashed(alloc, width, line_thickness),
        .underline_curly => try drawCurly(alloc, width, line_thickness),
        .overline => try drawSingle(alloc, width, line_thickness),
        .strikethrough => try drawSingle(alloc, width, line_thickness),
        else => unreachable,
    };
    defer canvas.deinit();

    // Write the drawing to the atlas
    const region = try canvas.writeAtlas(alloc, atlas);

    return font.Glyph{
        .width = width,
        .height = @intCast(region.height),
        .offset_x = 0,
        // Glyph.offset_y is the distance between the top of the glyph and the
        // bottom of the cell. We want the top of the glyph to be at line_pos
        // from the TOP of the cell, and then offset by the offset_y from the
        // draw function.
        .offset_y = @as(i32, @intCast(height -| line_pos)) - offset_y,
        .atlas_x = region.x,
        .atlas_y = region.y,
        .advance_x = @floatFromInt(width),
    };
}

/// A tuple with the canvas that the desired sprite was drawn on and
/// a recommended offset (+Y = down) to shift its Y position by, to
/// correct for underline styles with additional thickness.
const CanvasAndOffset = struct { font.sprite.Canvas, i32 };

/// Draw a single underline.
fn drawSingle(alloc: Allocator, width: u32, thickness: u32) !CanvasAndOffset {
    const height: u32 = thickness;
    var canvas = try font.sprite.Canvas.init(alloc, width, height);

    canvas.rect(.{
        .x = 0,
        .y = 0,
        .width = width,
        .height = thickness,
    }, .on);

    const offset_y: i32 = 0;

    return .{ canvas, offset_y };
}

/// Draw a double underline.
fn drawDouble(alloc: Allocator, width: u32, thickness: u32) !CanvasAndOffset {
    // Our gap between lines will be at least 2px.
    // (i.e. if our thickness is 1, we still have a gap of 2)
    const gap = @max(2, thickness);

    const height: u32 = thickness * 2 * gap;
    var canvas = try font.sprite.Canvas.init(alloc, width, height);

    canvas.rect(.{
        .x = 0,
        .y = 0,
        .width = width,
        .height = thickness,
    }, .on);

    canvas.rect(.{
        .x = 0,
        .y = thickness * 2,
        .width = width,
        .height = thickness,
    }, .on);

    const offset_y: i32 = -@as(i32, @intCast(thickness));

    return .{ canvas, offset_y };
}

/// Draw a dotted underline.
fn drawDotted(alloc: Allocator, width: u32, thickness: u32) !CanvasAndOffset {
    const height: u32 = thickness;
    var canvas = try font.sprite.Canvas.init(alloc, width, height);

    const dot_width = @max(thickness, 3);
    const dot_count = @max((width / dot_width) / 2, 1);
    const gap_width = try std.math.divCeil(u32, width -| (dot_count * dot_width), dot_count);
    var i: u32 = 0;
    while (i < dot_count) : (i += 1) {
        // Ensure we never go out of bounds for the rect
        const x = @min(i * (dot_width + gap_width), width - 1);
        const rect_width = @min(width - x, dot_width);
        canvas.rect(.{
            .x = @intCast(x),
            .y = 0,
            .width = rect_width,
            .height = thickness,
        }, .on);
    }

    const offset_y: i32 = 0;

    return .{ canvas, offset_y };
}

/// Draw a dashed underline.
fn drawDashed(alloc: Allocator, width: u32, thickness: u32) !CanvasAndOffset {
    const height: u32 = thickness;
    var canvas = try font.sprite.Canvas.init(alloc, width, height);

    const dash_width = width / 3 + 1;
    const dash_count = (width / dash_width) + 1;
    var i: u32 = 0;
    while (i < dash_count) : (i += 2) {
        // Ensure we never go out of bounds for the rect
        const x = @min(i * dash_width, width - 1);
        const rect_width = @min(width - x, dash_width);
        canvas.rect(.{
            .x = @intCast(x),
            .y = 0,
            .width = rect_width,
            .height = thickness,
        }, .on);
    }

    const offset_y: i32 = 0;

    return .{ canvas, offset_y };
}

/// Draw a curly underline. Thanks to Wez Furlong for providing
/// the basic math structure for this since I was lazy with the
/// geometry.
fn drawCurly(alloc: Allocator, width: u32, thickness: u32) !CanvasAndOffset {
    const float_width: f64 = @floatFromInt(width);
    // Because of we way we draw the undercurl, we end up making it around 1px
    // thicker than it should be, to fix this we just reduce the thickness by 1.
    //
    // We use a minimum thickness of 0.414 because this empirically produces
    // the nicest undercurls at 1px underline thickness; thinner tends to look
    // too thin compared to straight underlines and has artefacting.
    const float_thick: f64 = @max(0.414, @as(f64, @floatFromInt(thickness -| 1)));

    // Calculate the wave period for a single character
    //   `2 * pi...` = 1 peak per character
    //   `4 * pi...` = 2 peaks per character
    const wave_period = 2 * std.math.pi / float_width;

    // The full amplitude of the wave can be from the bottom to the
    // underline position. We also calculate our mid y point of the wave
    const half_amplitude = 1.0 / wave_period;
    const y_mid: f64 = half_amplitude + float_thick * 0.5 + 1;

    // This is used in calculating the offset curve estimate below.
    const offset_factor = @min(1.0, float_thick * 0.5 * wave_period) * @min(1.0, half_amplitude * wave_period);

    const height: u32 = @intFromFloat(@ceil(half_amplitude + float_thick + 1) * 2);

    var canvas = try font.sprite.Canvas.init(alloc, width, height);

    // follow Xiaolin Wu's antialias algorithm to draw the curve
    var x: u32 = 0;
    while (x < width) : (x += 1) {
        // We sample the wave function at the *middle* of each
        // pixel column, to ensure that it renders symmetrically.
        const t: f64 = (@as(f64, @floatFromInt(x)) + 0.5) * wave_period;
        // Use the slope at this location to add thickness to
        // the line on this column, counteracting the thinning
        // caused by the slope.
        //
        // This is not the exact offset curve for a sine wave,
        // but it's a decent enough approximation.
        //
        // How did I derive this? I stared at Desmos and fiddled
        // with numbers for an hour until it was good enough.
        const t_u: f64 = t + std.math.pi;
        const slope_factor_u: f64 = (@sin(t_u) * @sin(t_u) * offset_factor) / ((1.0 + @cos(t_u / 2) * @cos(t_u / 2) * 2) * wave_period);
        const slope_factor_l: f64 = (@sin(t) * @sin(t) * offset_factor) / ((1.0 + @cos(t / 2) * @cos(t / 2) * 2) * wave_period);

        const cosx: f64 = @cos(t);
        // This will be the center of our stroke.
        const y: f64 = y_mid + half_amplitude * cosx;

        // The upper pixel and lower pixel are
        // calculated relative to the center.
        const y_u: f64 = y - float_thick * 0.5 - slope_factor_u;
        const y_l: f64 = y + float_thick * 0.5 + slope_factor_l;
        const y_upper: u32 = @intFromFloat(@floor(y_u));
        const y_lower: u32 = @intFromFloat(@ceil(y_l));
        const alpha_u: u8 = @intFromFloat(@round(255 * (1.0 - @abs(y_u - @floor(y_u)))));
        const alpha_l: u8 = @intFromFloat(@round(255 * (1.0 - @abs(y_l - @ceil(y_l)))));

        // upper and lower bounds
        canvas.pixel(x, @min(y_upper, height - 1), @enumFromInt(alpha_u));
        canvas.pixel(x, @min(y_lower, height - 1), @enumFromInt(alpha_l));

        // fill between upper and lower bound
        var y_fill: u32 = y_upper + 1;
        while (y_fill < y_lower) : (y_fill += 1) {
            canvas.pixel(x, @min(y_fill, height - 1), .on);
        }
    }

    const offset_y: i32 = @intFromFloat(-@round(half_amplitude));

    return .{ canvas, offset_y };
}

test "single" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var atlas_grayscale = try font.Atlas.init(alloc, 512, .grayscale);
    defer atlas_grayscale.deinit(alloc);

    _ = try renderGlyph(
        alloc,
        &atlas_grayscale,
        .underline,
        36,
        18,
        9,
        2,
    );
}

test "strikethrough" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var atlas_grayscale = try font.Atlas.init(alloc, 512, .grayscale);
    defer atlas_grayscale.deinit(alloc);

    _ = try renderGlyph(
        alloc,
        &atlas_grayscale,
        .strikethrough,
        36,
        18,
        9,
        2,
    );
}

test "single large thickness" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var atlas_grayscale = try font.Atlas.init(alloc, 512, .grayscale);
    defer atlas_grayscale.deinit(alloc);

    // unrealistic thickness but used to cause a crash
    // https://github.com/mitchellh/ghostty/pull/1548
    _ = try renderGlyph(
        alloc,
        &atlas_grayscale,
        .underline,
        36,
        18,
        9,
        200,
    );
}

test "curly" {
    const testing = std.testing;
    const alloc = testing.allocator;

    var atlas_grayscale = try font.Atlas.init(alloc, 512, .grayscale);
    defer atlas_grayscale.deinit(alloc);

    _ = try renderGlyph(
        alloc,
        &atlas_grayscale,
        .underline_curly,
        36,
        18,
        9,
        2,
    );
}
