'use client'

/**
 * SectionControls.tsx
 * 
 * UI panel for controlling 3D-to-2D section extraction.
 * Allows user to:
 * - Enable/disable section view
 * - Choose slice axis (X or Y)
 * - Set slice position with slider
 * - Override UDL if desired
 * - View tributary width and calculated loads
 */

import { useDesignStore } from '@/store/designStore'

export default function SectionControls() {
  const {
    // 3D params for reference
    params,
    // Section state
    sectionMode,
    sliceAxis,
    slicePosition,
    udlOverride,
    sectionDeflectionScale,
    sectionResult,
    sectionLoading,
    sectionError,
    // Section actions
    setSectionMode,
    setSliceAxis,
    setSlicePosition,
    setUdlOverride,
    setSectionDeflectionScale,
  } = useDesignStore()

  // Calculate actual slice coordinate for display
  const maxCoord = sliceAxis === 'x' ? params.width : params.depth
  const sliceCoord = slicePosition * maxCoord
  const gridCount = sliceAxis === 'x' ? params.nx : params.ny
  const gridSpacing = maxCoord / (gridCount - 1)
  const gridIndex = Math.round(slicePosition * (gridCount - 1))

  return (
    <div className="section-controls">
      <div className="section-header">
        <h3>Section View</h3>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={sectionMode}
            onChange={(e) => setSectionMode(e.target.checked)}
          />
          <span className="toggle-slider"></span>
        </label>
      </div>

      {sectionMode && (
        <>
          {/* Slice Axis Selection */}
          <div className="control-group">
            <label>Slice Axis</label>
            <div className="axis-buttons">
              <button
                className={`axis-btn ${sliceAxis === 'x' ? 'active' : ''}`}
                onClick={() => setSliceAxis('x')}
              >
                X (Width)
              </button>
              <button
                className={`axis-btn ${sliceAxis === 'y' ? 'active' : ''}`}
                onClick={() => setSliceAxis('y')}
              >
                Y (Depth)
              </button>
            </div>
          </div>

          {/* Slice Position Slider */}
          <div className="control-group">
            <label>
              Position: {sliceAxis.toUpperCase()} = {sliceCoord.toFixed(2)} m
              <span className="hint"> (Grid {gridIndex + 1}/{gridCount})</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={slicePosition}
              onChange={(e) => setSlicePosition(parseFloat(e.target.value))}
              className="slider"
            />
            <div className="slider-labels">
              <span>0 m</span>
              <span>{maxCoord.toFixed(1)} m</span>
            </div>
          </div>

          {/* Deflection Scale */}
          <div className="control-group">
            <label>Deflection Scale: {sectionDeflectionScale}×</label>
            <input
              type="range"
              min={1}
              max={200}
              step={1}
              value={sectionDeflectionScale}
              onChange={(e) => setSectionDeflectionScale(parseFloat(e.target.value))}
              className="slider"
            />
          </div>

          {/* UDL Override */}
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={udlOverride !== null}
                onChange={(e) => setUdlOverride(e.target.checked ? 5.0 : null)}
              />
              {' '}Override UDL
            </label>
            {udlOverride !== null && (
              <div className="udl-input">
                <input
                  type="number"
                  min={0}
                  max={50}
                  step={0.5}
                  value={udlOverride}
                  onChange={(e) => setUdlOverride(parseFloat(e.target.value))}
                />
                <span>kN/m</span>
              </div>
            )}
          </div>

          {/* Section Info */}
          {sectionLoading && (
            <div className="section-info loading">
              <span className="spinner"></span> Extracting section...
            </div>
          )}

          {sectionError && (
            <div className="section-info error">
              ⚠️ {sectionError}
            </div>
          )}

          {sectionResult && sectionResult.success && (
            <div className="section-info success">
              <div className="info-row">
                <span>Slice:</span>
                <span>{sectionResult.slice_axis?.toUpperCase()} = {sectionResult.slice_value?.toFixed(2)} m</span>
              </div>
              <div className="info-row">
                <span>Nodes:</span>
                <span>{sectionResult.n_nodes_extracted}</span>
              </div>
              <div className="info-row">
                <span>Elements:</span>
                <span>{sectionResult.n_elements_extracted}</span>
              </div>
              <div className="info-row">
                <span>Span:</span>
                <span>{sectionResult.section_span?.toFixed(2)} m</span>
              </div>
              <div className="info-row highlight">
                <span>Tributary Width:</span>
                <span>{sectionResult.tributary_width?.toFixed(2)} m</span>
              </div>
              <div className="info-row">
                <span>Calc. UDL:</span>
                <span>{sectionResult.calculated_udl_kn_m?.toFixed(2)} kN/m</span>
              </div>
              <div className="info-row highlight">
                <span>Applied UDL:</span>
                <span>{sectionResult.applied_udl_kn_m?.toFixed(2)} kN/m</span>
              </div>

              {/* Metrics from frame result */}
              {sectionResult.frame_result?.metrics && (
                <>
                  <hr />
                  <div className="info-row">
                    <span>Max Disp:</span>
                    <span>{sectionResult.frame_result.metrics.max_displacement_mm.toFixed(2)} mm</span>
                  </div>
                  <div className="info-row">
                    <span>Drift:</span>
                    <span>{sectionResult.frame_result.metrics.drift_mm.toFixed(2)} mm</span>
                  </div>
                  <div className="info-row">
                    <span>Max M:</span>
                    <span>{(sectionResult.frame_result.metrics.max_moment / 1000).toFixed(2)} kN·m</span>
                  </div>
                  <div className="info-row">
                    <span>Max V:</span>
                    <span>{(sectionResult.frame_result.metrics.max_shear_force / 1000).toFixed(2)} kN</span>
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}

      <style jsx>{`
        .section-controls {
          padding: 12px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          margin-top: 12px;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }

        .section-header h3 {
          margin: 0;
          font-size: 14px;
          color: #e0e0e0;
        }

        .toggle-switch {
          position: relative;
          display: inline-block;
          width: 44px;
          height: 24px;
        }

        .toggle-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }

        .toggle-slider {
          position: absolute;
          cursor: pointer;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: #444;
          transition: 0.3s;
          border-radius: 24px;
        }

        .toggle-slider:before {
          position: absolute;
          content: "";
          height: 18px;
          width: 18px;
          left: 3px;
          bottom: 3px;
          background-color: white;
          transition: 0.3s;
          border-radius: 50%;
        }

        input:checked + .toggle-slider {
          background-color: #4CAF50;
        }

        input:checked + .toggle-slider:before {
          transform: translateX(20px);
        }

        .control-group {
          margin-bottom: 12px;
        }

        .control-group label {
          display: block;
          font-size: 12px;
          color: #aaa;
          margin-bottom: 4px;
        }

        .hint {
          color: #666;
          font-size: 11px;
        }

        .axis-buttons {
          display: flex;
          gap: 8px;
        }

        .axis-btn {
          flex: 1;
          padding: 8px;
          background: #333;
          border: 1px solid #555;
          border-radius: 4px;
          color: #aaa;
          cursor: pointer;
          font-size: 12px;
          transition: all 0.2s;
        }

        .axis-btn:hover {
          background: #444;
        }

        .axis-btn.active {
          background: #4CAF50;
          border-color: #4CAF50;
          color: white;
        }

        .slider {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: #333;
          outline: none;
          -webkit-appearance: none;
        }

        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #4CAF50;
          cursor: pointer;
        }

        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 10px;
          color: #666;
          margin-top: 2px;
        }

        .udl-input {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
        }

        .udl-input input {
          width: 80px;
          padding: 6px;
          background: #333;
          border: 1px solid #555;
          border-radius: 4px;
          color: white;
          font-size: 12px;
        }

        .udl-input span {
          color: #888;
          font-size: 12px;
        }

        .section-info {
          padding: 10px;
          border-radius: 6px;
          font-size: 12px;
          margin-top: 12px;
        }

        .section-info.loading {
          background: rgba(33, 150, 243, 0.2);
          color: #90CAF9;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .spinner {
          width: 14px;
          height: 14px;
          border: 2px solid #90CAF9;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .section-info.error {
          background: rgba(244, 67, 54, 0.2);
          color: #EF9A9A;
        }

        .section-info.success {
          background: rgba(76, 175, 80, 0.1);
          border: 1px solid rgba(76, 175, 80, 0.3);
        }

        .info-row {
          display: flex;
          justify-content: space-between;
          padding: 3px 0;
          color: #aaa;
        }

        .info-row.highlight {
          color: #4CAF50;
          font-weight: 500;
        }

        .info-row span:last-child {
          font-family: monospace;
        }

        hr {
          border: none;
          border-top: 1px solid #444;
          margin: 8px 0;
        }
      `}</style>
    </div>
  )
}

