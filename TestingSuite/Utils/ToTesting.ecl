EXPORT ToTesting(Resdata, dOut) := MACRO
    IMPORT Std;
    LOADXML('<xml/>');
    #EXPORTXML(fields, RECORDOF(Resdata));
		
		#DECLARE(OutStr);
		#SET(OutStr,'{');

		#FOR(fields)
				#FOR(Field)
					#IF (%'{@name}'% = 'id')
						#APPEND(OutStr, %'{@name}'%);
						#APPEND(OutStr, ',');
					#ELSEIF (%'{@name}'% = 'class')
						#APPEND(OutStr, %'{@name}'%);
						#APPEND(OutStr, ',');
					#END
				#END
		#END
		#APPEND(OutStr, '}');
		dOut := TABLE(Resdata, #EXPAND(%'OutStr'%));
ENDMACRO;