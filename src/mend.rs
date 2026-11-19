//! Lines the tracing left in pieces, joined: the one stage that is not Kraken's.
//!
//! Kraken makes a line of every blob of the baseline map. Where the network
//! is unsure of a line — faint pencil, a wide gap between two words — the map
//! breaks, and one written line comes back as two or three. On handwriting
//! that is one line in thirty.
//!
//! Two pieces that lie end to end are not always one line, though: the two
//! halves of a verse do, and so does a line and the line beside it on the
//! facing page. What tells them apart is something Kraken reads only to turn
//! a line the right way round. The network marks where a line starts and
//! where it ends, and marks it firmly — on print nineteen ends in twenty —
//! while the two ends that face each other across a break have no mark at
//! all. So pieces are joined where they lie in a line *and* neither facing
//! end is marked. Joining by position alone mends handwriting but merges
//! everything set in two columns.

/// A point of the heatmap as (x, y).
pub type Point = [i32; 2];

/// How far apart two pieces of one line may lie, in the page's line spacings.
/// Results barely change between 1.5 and 6: the markers decide, and this only
/// keeps a line from being joined to something across the page.
const GAP: f64 = 4.0;
/// How far to the side of where a piece is heading the next may begin, in
/// line spacings: a third, or it is the line above or below.
const ASIDE: f64 = 0.35;
/// How far the two pieces' headings may differ, in degrees.
const TURN: f64 = 15.0;
/// A start or end marker this sure says a line really ends here. A true end
/// is marked near 1 and a break not at all; there is little in between.
const MARKED: f32 = 0.3;
/// The spacing taken for a page with too few lines to measure one, as a
/// share of the page's height.
const USUAL_SPACING: f64 = 0.02;

/// A traced line, and what the network marked at its ends.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub points: Vec<Point>,
    /// At the first point and at the last: how sure the start marker is
    /// there, and the end marker.
    pub marks: [[f32; 2]; 2],
    /// The traced lines this one was joined from, as they were: empty for a
    /// line as traced. Kept because the joining can be taken back — a line
    /// that cannot be given an outline is worth less than its pieces.
    pub of: Vec<Vec<Point>>,
}

type V = [f64; 2];

fn at(p: Point) -> V {
    [p[0] as f64, p[1] as f64]
}

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn unit(v: V) -> V {
    let n = v[0].hypot(v[1]);
    if n > 0.0 {
        [v[0] / n, v[1] / n]
    } else {
        v
    }
}

/// Which way a line is heading as it arrives at an end, judged over its
/// last `reach` pixels: a line's last stretch alone may be a jog of the
/// tracing.
fn heading(points: &[Point], at_last: bool, reach: f64) -> V {
    let n = points.len();
    let nth = |k: usize| at(if at_last { points[n - 1 - k] } else { points[k] });
    let tip = nth(0);
    let mut gone = 0.0;
    for k in 1..n {
        let step = sub(nth(k - 1), nth(k));
        gone += step[0].hypot(step[1]);
        if gone >= reach || k == n - 1 {
            return unit(sub(tip, nth(k)));
        }
    }
    [0.0, 0.0]
}

/// How far `p` is from a line, if the nearest of the line is not one of its
/// ends: a line that lies above or below `p`, not one further along.
fn beside(p: V, line: &[Point]) -> Option<f64> {
    let mut best = (f64::MAX, false);
    for (k, pair) in line.windows(2).enumerate() {
        let (a, b) = (at(pair[0]), at(pair[1]));
        let ab = sub(b, a);
        let along = if dot(ab, ab) == 0.0 {
            0.0
        } else {
            dot(sub(p, a), ab) / dot(ab, ab)
        };
        let within = (along > 0.0 && along < 1.0) || (along <= 0.0 && k > 0) || (along >= 1.0 && k + 2 < line.len());
        let t = along.clamp(0.0, 1.0);
        let d = sub(p, [a[0] + t * ab[0], a[1] + t * ab[1]]);
        let d = d[0].hypot(d[1]);
        if d < best.0 {
            best = (d, within);
        }
    }
    best.1.then_some(best.0)
}

/// The page's line spacing: the usual distance from the middle of a line to
/// the nearest line beside it.
fn spacing(pieces: &[Piece], page_height: usize) -> f64 {
    let mut gaps: Vec<f64> = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        let n = piece.points.len();
        let middle = if n > 2 {
            at(piece.points[n / 2])
        } else {
            let (a, b) = (at(piece.points[0]), at(piece.points[n - 1]));
            [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
        };
        let nearest = pieces
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .filter_map(|(_, other)| beside(middle, &other.points))
            // Closer than this it is the same line traced twice, not the next.
            .filter(|&d| d > 3.0)
            .fold(f64::MAX, f64::min);
        if nearest < f64::MAX {
            gaps.push(nearest);
        }
    }
    if gaps.len() < 3 {
        return USUAL_SPACING * page_height as f64;
    }
    gaps.sort_by(f64::total_cmp);
    let n = gaps.len();
    if n % 2 == 1 {
        gaps[n / 2]
    } else {
        (gaps[n / 2 - 1] + gaps[n / 2]) / 2.0
    }
}

/// Whether an end of a piece is a line's end by the network's own word.
fn marked(piece: &Piece, at_last: bool) -> bool {
    let [start, end] = piece.marks[at_last as usize];
    start.max(end) >= MARKED
}

/// How far `b` begins from where `a` ends, if `b` carries `a` on: `a` taken
/// from its last point if `a_last`, `b` from its first if `b_first`.
fn carries_on(a: &Piece, a_last: bool, b: &Piece, b_first: bool, spacing: f64) -> Option<f64> {
    if marked(a, a_last) || marked(b, !b_first) {
        return None;
    }
    let end = |piece: &Piece, last: bool| {
        at(if last {
            piece.points[piece.points.len() - 1]
        } else {
            piece.points[0]
        })
    };
    let (from, to) = (end(a, a_last), end(b, !b_first));
    let gap = sub(to, from);
    let far = gap[0].hypot(gap[1]);
    if far > GAP * spacing {
        return None;
    }
    // Both as headings from `a` into `b`.
    let out_of_a = heading(&a.points, a_last, 2.0 * spacing);
    let into_b = heading(&b.points, !b_first, 2.0 * spacing);
    let into_b = [-into_b[0], -into_b[1]];
    if dot(out_of_a, into_b).clamp(-1.0, 1.0).acos().to_degrees() > TURN {
        return None;
    }
    // `b` lies ahead of `a`, not beside it and not back along it.
    let (ahead, aside) = (dot(gap, out_of_a), (out_of_a[0] * gap[1] - out_of_a[1] * gap[0]).abs());
    if ahead < -0.25 * spacing || aside > ASIDE * spacing {
        return None;
    }
    (dot(sub(end(b, b_first), from), out_of_a) > 0.0).then_some(far)
}

/// The pieces with those that are one line joined, nearest ends first, until
/// nothing more joins. `page_height` is the heatmap's, and `the_other_way`
/// says of a line's two ends' markers and where the ends lie whether the
/// line runs backwards — the tracing's own rule, applied again to a line
/// that has new ends.
pub fn joined(mut pieces: Vec<Piece>, page_height: usize, the_other_way: impl Fn(&Piece) -> bool) -> Vec<Piece> {
    pieces.retain(|piece| piece.points.len() >= 2);
    let spacing = spacing(&pieces, page_height);
    loop {
        let mut nearest: Option<(f64, usize, bool, usize, bool)> = None;
        for (i, a) in pieces.iter().enumerate() {
            for (j, b) in pieces.iter().enumerate() {
                if i == j {
                    continue;
                }
                for (a_last, b_first) in [(true, true), (true, false), (false, true), (false, false)] {
                    if let Some(far) = carries_on(a, a_last, b, b_first, spacing) {
                        if nearest.is_none_or(|(best, ..)| far < best) {
                            nearest = Some((far, i, a_last, j, b_first));
                        }
                    }
                }
            }
        }
        let Some((_, i, a_last, j, b_first)) = nearest else {
            return pieces;
        };
        let b = pieces[j].clone();
        let a = &mut pieces[i];
        if a.of.is_empty() {
            a.of.push(a.points.clone());
        }
        a.of.extend(if b.of.is_empty() { vec![b.points.clone()] } else { b.of.clone() });
        if !a_last {
            a.points.reverse();
            a.marks.reverse();
        }
        if b_first {
            a.points.extend(&b.points);
            a.marks[1] = b.marks[1];
        } else {
            a.points.extend(b.points.iter().rev());
            a.marks[1] = b.marks[0];
        }
        if the_other_way(a) {
            a.points.reverse();
            a.marks.reverse();
        }
        pieces.remove(j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNMARKED: [f32; 2] = [0.0, 0.0];
    const STARTS: [f32; 2] = [1.0, 0.0];
    const ENDS: [f32; 2] = [0.0, 1.0];

    fn piece(points: &[Point], first: [f32; 2], last: [f32; 2]) -> Piece {
        Piece {
            points: points.to_vec(),
            marks: [first, last],
            of: Vec::new(),
        }
    }

    /// Lines forty pixels apart down a page, to measure a spacing by.
    fn a_page_of(mut pieces: Vec<Piece>) -> Vec<Piece> {
        for row in 1..=4 {
            let y = 100 + 40 * row;
            pieces.push(piece(&[[600, y], [100, y]], STARTS, ENDS));
        }
        pieces
    }

    fn as_it_lies(_: &Piece) -> bool {
        false
    }

    #[test]
    fn two_pieces_end_to_end_with_no_marker_between_them_are_one_line() {
        let pieces = a_page_of(vec![
            piece(&[[600, 100], [400, 101]], STARTS, UNMARKED),
            piece(&[[370, 101], [100, 100]], UNMARKED, ENDS),
        ]);

        let lines = joined(pieces, 1800, as_it_lies);

        assert_eq!(lines.len(), 5);
        let whole = lines.iter().find(|l| l.points.len() == 4).unwrap();
        assert_eq!(whole.points, vec![[600, 100], [400, 101], [370, 101], [100, 100]]);
        assert_eq!(whole.marks, [STARTS, ENDS]);
        // And what it was made of is kept, for taking the joining back.
        assert_eq!(whole.of, vec![vec![[600, 100], [400, 101]], vec![[370, 101], [100, 100]]]);
    }

    #[test]
    fn the_halves_of_a_verse_each_marked_where_it_ends_stay_two() {
        let pieces = a_page_of(vec![
            piece(&[[600, 100], [400, 100]], STARTS, ENDS),
            piece(&[[370, 100], [100, 100]], STARTS, ENDS),
        ]);

        assert_eq!(joined(pieces, 1800, as_it_lies).len(), 6);
    }

    #[test]
    fn one_marked_end_is_enough_to_keep_two_lines_apart() {
        let pieces = a_page_of(vec![
            piece(&[[600, 100], [400, 100]], STARTS, ENDS),
            piece(&[[370, 100], [100, 100]], UNMARKED, ENDS),
        ]);

        assert_eq!(joined(pieces, 1800, as_it_lies).len(), 6);
    }

    #[test]
    fn a_line_and_the_line_under_it_are_not_joined() {
        // The second begins under the first's end, a line spacing down.
        let pieces = a_page_of(vec![
            piece(&[[600, 60], [400, 60]], UNMARKED, UNMARKED),
            piece(&[[390, 100], [100, 100]], UNMARKED, UNMARKED),
        ]);

        assert_eq!(joined(pieces, 1800, as_it_lies).len(), 6);
    }

    #[test]
    fn a_piece_heading_off_at_an_angle_is_another_line() {
        let pieces = a_page_of(vec![
            piece(&[[600, 100], [400, 100]], UNMARKED, UNMARKED),
            piece(&[[380, 102], [250, 40]], UNMARKED, UNMARKED),
        ]);

        assert_eq!(joined(pieces, 1800, as_it_lies).len(), 6);
    }

    #[test]
    fn pieces_across_the_page_from_each_other_are_not_joined() {
        // Ten line spacings of nothing between them.
        let pieces = a_page_of(vec![
            piece(&[[900, 100], [800, 100]], UNMARKED, UNMARKED),
            piece(&[[400, 100], [100, 100]], UNMARKED, UNMARKED),
        ]);

        assert_eq!(joined(pieces, 1800, as_it_lies).len(), 6);
    }

    #[test]
    fn three_pieces_of_a_line_are_joined_into_one_whichever_way_each_was_traced() {
        let pieces = a_page_of(vec![
            piece(&[[600, 100], [450, 100]], STARTS, UNMARKED),
            // Traced the other way round, as an unmarked piece is.
            piece(&[[300, 100], [430, 100]], UNMARKED, UNMARKED),
            piece(&[[280, 100], [100, 100]], UNMARKED, ENDS),
        ]);

        let lines = joined(pieces, 1800, as_it_lies);

        assert_eq!(lines.len(), 5);
        let whole = lines.iter().find(|l| l.points.len() == 6).unwrap();
        assert_eq!(whole.of.len(), 3);
        let xs: Vec<i32> = whole.points.iter().map(|p| p[0]).collect();
        assert!(xs == vec![600, 450, 430, 300, 280, 100] || xs == vec![100, 280, 300, 430, 450, 600]);
    }

    #[test]
    fn a_joined_line_is_turned_as_the_tracing_would_turn_it() {
        let pieces = a_page_of(vec![
            piece(&[[100, 100], [300, 100]], UNMARKED, UNMARKED),
            piece(&[[330, 100], [600, 100]], UNMARKED, UNMARKED),
        ]);

        let lines = joined(pieces, 1800, |line| line.points[0][0] < line.points[line.points.len() - 1][0]);

        let whole = lines.iter().find(|l| l.points.len() == 4).unwrap();
        assert_eq!(whole.points[0], [600, 100]);
    }

    #[test]
    fn a_page_with_too_few_lines_to_measure_is_taken_to_be_spaced_as_pages_are() {
        let one = vec![piece(&[[600, 100], [400, 100]], UNMARKED, UNMARKED)];

        assert_eq!(spacing(&one, 1800), 36.0);
    }
}
