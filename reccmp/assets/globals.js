/** @import {ReccmpSerializedReport, ReccmpComparedEntity} from './types' */

/**
 * @typedef {object} ReccmpWindowProps
 * @property {ReccmpSerializedReport} global_reccmp_report
 */

// reccmp-pack-begin

/** @type {Window & ReccmpWindowProps} */
const reccmpWindow = /** @type {?} */ (window);

const { data: global_reccmp_data, ...global_reccmp_metadata } = reccmpWindow.global_reccmp_report;

// Unwrap array of functions into a dictionary with address as the key.
const dataDict = Object.fromEntries(
  global_reccmp_data.map(
    /**
     * @param {ReccmpComparedEntity} row
     * @returns {[string, ReccmpComparedEntity]}
     */
    (row) => [row.address, row],
  ),
);

/**
 * Current reports carry the verifier status in ``comparison``; reports from
 * before that field only have the boolean ``effective``.
 * @param {ReccmpComparedEntity} row
 * @returns {boolean}
 */
function isEffectiveMatch(row) {
  if (row.comparison !== undefined) {
    return row.comparison.status === 'effective';
  }
  return row.effective === true;
}

/**
 * @param {string} addr
 * @returns {ReccmpComparedEntity}
 */
function getDataByAddr(addr) {
  return dataDict[addr];
}

// reccmp-pack-end

export { global_reccmp_data, global_reccmp_metadata, getDataByAddr, isEffectiveMatch };
