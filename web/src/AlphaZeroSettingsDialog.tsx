import { makeStyles, tokens } from '@fluentui/react-components';

import IntInput from './IntInput';
import { AlphaZeroSettings } from './PlayerSettings';

export default function AlphaZeroSettingsDialog(
  { settings, onChange }: AlphaZeroSettingsDialogProps
) {
  const styles = useStyles();

  function validateNumSimulations(value: number) {
    if (value < 1) {
      onChange(null);
      return 'Number of simulations must be greater than or equal to 1.';
    }
    onChange({ ...settings, numSimulations: value });
    return null;
  }

  function validateBatchSize(value: number) {
    if (value < 1) {
      onChange(null);
      return 'Batch size must be greater than or equal to 1.';
    }
    onChange({ ...settings, batchSize: value });
    return null;
  }

  return (
    <div className={styles.root}>
      <IntInput
        label='Number of Simulations'
        value={settings.numSimulations}
        validate={validateNumSimulations}
      />
      <IntInput
        label='Batch Size'
        value={settings.batchSize}
        validate={validateBatchSize}
      />
    </div>
  );
}

export type AlphaZeroSettingsDialogProps = {
  settings: AlphaZeroSettings;
  onChange: (settings: AlphaZeroSettings | null) => void;
};

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'start',
    gap: tokens.spacingVerticalM,
  },
});
