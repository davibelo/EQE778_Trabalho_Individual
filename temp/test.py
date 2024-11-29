import os
import win32com.client as win32

file = r"UTAA_revK.bkp"
aspen_Path = os.path.abspath(file)

print('Connecting to the Aspen Plus... Please wait ')
Application = win32.Dispatch('Apwn.Document') # Registered name of Aspen Plus
print('Connected!')

Application.InitFromArchive2(aspen_Path)
Application.visible = 0

# Read variables
test_temp = Application.Tree.FindNode(r"\Data\Streams\G-ACID\Output\TEMP_OUT\MIXED").Value
print(f"temperature: {test_temp} °C")

test_mass_frac = Application.Tree.FindNode(r"\Data\Streams\AGUAPR1\Output\MASSFRAC\MIXED\H2S").Value
print(f"Mass Fraction H2S before: {test_mass_frac}")
print(type(test_mass_frac))

test_reboiler_duty = Application.Tree.FindNode(r"\Data\Blocks\T1\Input\QN").Value
print(f"Reboiler Duty before: {test_reboiler_duty}")

# Change variables
new_reboiler_duty = test_reboiler_duty + 20000
print(f"Modifying Reboiler Duty to {new_reboiler_duty}")
Application.Tree.FindNode(r"\Data\Blocks\T1\Input\QN").Value = new_reboiler_duty

# Run
Application.Engine.Run2()

# Show results
test_mass_frac = Application.Tree.FindNode(r"\Data\Streams\AGUAPR1\Output\MASSFRAC\MIXED\H2S").Value
print(f"Mass Fraction H2S: {test_mass_frac}")

# Close the Aspen simulation
Application.Close()
