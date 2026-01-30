"use client"

import Slider from '@mui/material/Slider'
import { styled } from '@mui/material/styles'

interface RangeSliderProps {
  minValue: number
  maxValue: number
  onMinChange: (value: number) => void
  onMaxChange: (value: number) => void
  label?: string
}

// Styled MUI Slider to match your design
const StyledSlider = styled(Slider)({
  color: '#3b82f6', // Blue color
  height: 6,
  '& .MuiSlider-track': {
    border: 'none',
  },
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
      boxShadow: 'inherit',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-valueLabel': {
    lineHeight: 1.2,
    fontSize: 12,
    background: 'unset',
    padding: 0,
    width: 32,
    height: 32,
    borderRadius: '50% 50% 50% 0',
    backgroundColor: '#3b82f6',
    transformOrigin: 'bottom left',
    transform: 'translate(50%, -100%) rotate(-45deg) scale(0)',
    '&:before': { display: 'none' },
    '&.MuiSlider-valueLabelOpen': {
      transform: 'translate(50%, -100%) rotate(-45deg) scale(1)',
    },
    '& > *': {
      transform: 'rotate(45deg)',
    },
  },
})

export function RangeSlider({
  minValue,
  maxValue,
  onMinChange,
  onMaxChange,
  label = "Probability Constraint Range (0-100%)"
}: RangeSliderProps) {

  const minDistance = 1

  const handleChange = (
    event: Event,
    newValue: number | number[],
    activeThumb: number
  ) => {
    if (!Array.isArray(newValue)) {
      return
    }

    if (activeThumb === 0) {
      // MIN thumb moved - ensure it doesn't get too close to MAX
      const newMin = Math.min(newValue[0], maxValue - minDistance)
      onMinChange(newMin)
    } else {
      // MAX thumb moved - ensure it doesn't get too close to MIN
      const newMax = Math.max(newValue[1], minValue + minDistance)
      onMaxChange(newMax)
    }
  }

  return (
    <div className="mt-4 sm:mt-6">
      <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3">{label}</p>
      <div className="flex items-center gap-2 sm:gap-4">
        <span className="text-gray-400 text-xs sm:text-sm">0</span>
        <div className="flex-1">
          <StyledSlider
            value={[minValue, maxValue]}
            onChange={handleChange}
            min={0}
            max={100}
            step={1}
            disableSwap
          />
          <div className="flex justify-between mt-1 px-1">
            <span className="text-gray-500 text-xs">MIN: {minValue}%</span>
            <span className="text-gray-500 text-xs">MAX: {maxValue}%</span>
          </div>
        </div>
        <span className="text-gray-400 text-xs sm:text-sm w-8 sm:w-10">100</span>
      </div>
    </div>
  )
}
