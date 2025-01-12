import { Input, Label, Text, makeStyles, tokens, useId } from '@fluentui/react-components';
import { CheckmarkCircleFilled, DismissCircleFilled } from '@fluentui/react-icons';
import React, { useState } from 'react';

export default function IntInput({ label, value, validate }: IntInputProps) {
  const styles = useStyles();

  const inputId = useId();
  const [inputValue, setInputValue] = useState(String(value));
  const [error, setError] = useState<string | null>(null);

  function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    setInputValue(event.target.value);
    const parsedValue = parseInt(event.target.value, 10);
    if (Number.isInteger(parsedValue) && String(parsedValue) === event.target.value) {
      setError(validate(parsedValue));
    } else {
      setError('Value must be an integer.');
    }
  }

  return (
    <div className={styles.root}>
      <Label htmlFor={inputId}>{`${label}:`}</Label>
      <Input
        id={inputId}
        type='number'
        value={inputValue}
        onChange={handleChange}
        contentAfter={
          error === null
            ? <CheckmarkCircleFilled className={styles.checkmarkIcon} />
            : <DismissCircleFilled className={styles.dismissIcon} />
        }
      ></Input>
      <div className={styles.errorContainer}>
        <Text className={styles.error}>{error ?? ''}</Text>
      </div>
    </div>
  );
}

export type IntInputProps = {
  label: string;
  value: number;
  validate: (value: number) => string | null;
};

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'start',
  },
  checkmarkIcon: {
    color: tokens.colorPaletteGreenForeground3,
  },
  dismissIcon: {
    color: tokens.colorPaletteRedForeground3,
  },
  errorContainer: {
    height: '1.5em',
  },
  error: {
    color: tokens.colorPaletteRedForeground2,
  },
});
